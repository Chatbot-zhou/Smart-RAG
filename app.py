import asyncio
import json
import re
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from base.config import single_config as config
from new_main import IntegratedQASystem


app = FastAPI(title="EduRAG API", description="集成 MySQL FAQ 和 RAG 的智能问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_qa_system: IntegratedQASystem | None = None
_qa_init_error: str | None = None


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


GREETING_PATTERNS = [
    (r"^(你好|您好|hi|hello)", "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"),
    (r"^(你是谁|您是谁|你叫什么|你的名字|who are you)", "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"),
    (r"^(在吗|在不在|有人吗)", "我在！我是黑马程序员，随时为你解答问题！"),
    (r"^(干嘛呢|你在干嘛|做什么)", "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"),
]


def get_qa_system() -> IntegratedQASystem:
    global _qa_system, _qa_init_error
    if _qa_system is not None:
        return _qa_system
    try:
        _qa_system = IntegratedQASystem()
        _qa_init_error = None
        return _qa_system
    except Exception as exc:
        _qa_init_error = str(exc)
        raise HTTPException(status_code=503, detail=f"问答系统初始化失败: {_qa_init_error}") from exc


def check_greeting(query: str) -> str | None:
    query_text = query.strip()
    for pattern, response in GREETING_PATTERNS:
        if re.match(pattern, query_text, re.IGNORECASE):
            return response
    return None


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "qa_ready": _qa_system is not None,
        "qa_init_error": _qa_init_error,
    }


@app.get("/api/sources")
async def get_sources():
    return {"sources": config.VALID_SOURCES}


@app.post("/api/create_session")
async def create_session():
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    qa_system = get_qa_system()
    return {"session_id": session_id, "history": qa_system.get_session_history(session_id)}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    qa_system = get_qa_system()
    if qa_system.clear_session_history(session_id):
        return {"status": "success", "message": "历史记录已清除"}
    raise HTTPException(status_code=500, detail="清除历史记录失败")


@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }

    qa_system = get_qa_system()
    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        return {
            "answer": "请使用 WebSocket 接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
        }

    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time,
    }


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query_text = request_data.get("query", "")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id") or str(uuid.uuid4())
            start_time = time.time()

            await websocket.send_json({"type": "start", "session_id": session_id})

            greeting_response = check_greeting(query_text)
            if greeting_response:
                await websocket.send_json({"type": "token", "token": greeting_response, "session_id": session_id})
                await websocket.send_json(
                    {
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time,
                    }
                )
                continue

            try:
                qa_system = get_qa_system()
            except HTTPException as exc:
                await websocket.send_json({"type": "error", "error": exc.detail})
                continue

            for token, is_complete in qa_system.query(query_text, source_filter=source_filter, session_id=session_id):
                if token:
                    await websocket.send_json({"type": "token", "token": token, "session_id": session_id})
                if is_complete:
                    await websocket.send_json(
                        {
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        }
                    )
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "error": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=10000, reload=False)
