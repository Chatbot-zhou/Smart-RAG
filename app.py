import asyncio
import json
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from legal_brain.agent.controller import agent_controller
from legal_brain.config import settings
from legal_brain.app_service import SmartLegalBrain
from legal_brain.schemas import (
    AgentPlaceholderResponse,
    ContractDraftRequest,
    ContractReviewRequest,
    IntentRequest,
    IntentResponse,
    QueryRequest,
)


app = FastAPI(title=f"{settings.app_name} API", description="面向中国大陆公司法务场景的法律 RAG 与 Agent 接口")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_brain: SmartLegalBrain | None = None
_brain_init_error: str | None = None


GREETING_PATTERNS = [
    (r"^(你好|您好|hi|hello)", f"您好，我是{settings.app_name}，可以协助进行法律问答、合同审查和合同制作。"),
    (r"^(你是谁|您是谁|你叫什么|你的名字|who are you)", f"我是{settings.app_name}，面向企业法务场景的智能助手。"),
    (r"^(在吗|在不在|有人吗)", "我在。请描述您的法律问题、合同审查需求或合同制作需求。"),
]


def get_brain() -> SmartLegalBrain:
    global _brain, _brain_init_error
    if _brain is not None:
        return _brain
    try:
        _brain = SmartLegalBrain()
        _brain_init_error = None
        return _brain
    except Exception as exc:
        _brain_init_error = str(exc)
        raise HTTPException(status_code=503, detail=f"{settings.app_name} 初始化失败: {_brain_init_error}") from exc


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
        "app_name": settings.app_name,
        "brain_ready": _brain is not None,
        "brain_init_error": _brain_init_error,
    }


@app.get("/api/sources")
async def get_sources():
    return {"sources": settings.legal_domains}


@app.post("/api/create_session")
async def create_session():
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    brain = get_brain()
    return {"session_id": session_id, "history": brain.get_session_history(session_id)}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    brain = get_brain()
    if brain.clear_session_history(session_id):
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
            "references": [],
            "disclaimer": settings.service_disclaimer,
        }

    return {
        "answer": "请使用 WebSocket 接口获取流式法律问答结果。",
        "is_streaming": True,
        "session_id": session_id,
        "processing_time": time.time() - start_time,
        "references": [],
        "disclaimer": settings.service_disclaimer,
    }


@app.post("/api/agent/intent", response_model=IntentResponse)
async def detect_agent_intent(request: IntentRequest):
    return agent_controller.route(request.text)


@app.post("/api/contracts/review", response_model=AgentPlaceholderResponse)
async def review_contract(_: ContractReviewRequest):
    return agent_controller.contract_review_placeholder()


@app.post("/api/contracts/draft", response_model=AgentPlaceholderResponse)
async def draft_contract(_: ContractDraftRequest):
    return agent_controller.contract_drafting_placeholder()


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            request_data = json.loads(await websocket.receive_text())
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
                        "disclaimer": settings.service_disclaimer,
                    }
                )
                continue

            try:
                brain = get_brain()
            except HTTPException as exc:
                await websocket.send_json({"type": "error", "error": exc.detail})
                continue

            for token, is_complete in brain.query(query_text, source_filter=source_filter, session_id=session_id):
                if token:
                    await websocket.send_json({"type": "token", "token": token, "session_id": session_id})
                if is_complete:
                    await websocket.send_json(
                        {
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                            "disclaimer": settings.service_disclaimer,
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

