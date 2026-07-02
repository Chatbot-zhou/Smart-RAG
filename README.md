# EduRAG 问答系统

这是一个源码最小化后的 RAG 项目，根目录就是项目根目录。系统由 FastAPI 服务、MySQL FAQ 检索、Redis 缓存、Milvus 向量检索和 DashScope 兼容 OpenAI API 组成。

## 目录

- `app.py`: FastAPI HTTP/WebSocket 服务入口
- `new_main.py`: MySQL FAQ 与 RAG 流程编排
- `base/`: 配置与日志
- `mysql_qa/`: FAQ 数据库、Redis 缓存和 BM25 检索
- `rag_qa/`: 文档处理、向量库、查询分类和检索增强生成
- `static/`: 前端页面

## 配置

复制 `.env.example` 为 `.env`，填写本机配置：

```powershell
Copy-Item .env.example .env
```

必须配置 `DASHSCOPE_API_KEY`。旧代码中的真实 Key 已从源码移除；那个 Key 已经落盘过，建议在 DashScope 控制台轮换。

模型不再随源码保存。默认会使用模型名加载，也可以通过环境变量改成本地路径：

- `EMBEDDING_MODEL`
- `RERANKER_MODEL`
- `BERT_PRETRAINED_MODEL`
- `QUERY_CLASSIFIER_MODEL`
- `DOCUMENT_SEGMENTATION_MODEL`

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 10000
```

访问 `http://127.0.0.1:10000/`。

首页和健康检查不依赖 MySQL、Redis、Milvus 或模型立即可用；真正问答接口会在首次调用时初始化完整系统。如果配置或外部服务缺失，会返回清晰的初始化错误。

## 数据

源码仓库不保留样例 PDF、DOCX、CSV、模型权重、训练结果或日志。需要写入 Milvus 时，把文档目录配置到 `DOCUMENT_DATA_DIR` 后运行：

```powershell
python -m rag_qa.core.vector_store
```

需要初始化 FAQ 数据时，把 CSV 配置到 `FAQ_CSV_PATH` 后运行：

```powershell
python -m mysql_qa.db.mysql_client
```
