# 智慧法务大脑

智慧法务大脑是面向企业法务场景的法律 RAG 与合同 Agent 原型。当前阶段聚焦法律问答主链路：BERT 意图分类、Redis/MySQL FAQ 优先命中、BM25 + BGE-M3 + Milvus 混合检索、ReRank 精排、LangChain 编排、Qwen 生成和 RAGAS 评估。

## 能力

- FAQ 优先命中：MySQL 存 FAQ 与问答记录，Redis 存 7 天 TTL 热点索引。
- 法律 RAG：官方法规知识库、BGE-M3 稠密+稀疏向量、Milvus hybrid search、ReRank 精排。
- LangChain 编排：query rewrite、PromptTemplate、answer chain。
- RAGAS 评估：faithfulness、answer relevancy、context precision、context recall。
- Agent 骨架：合同审查、合同制作接口保留到下一阶段扩展。

## 配置

复制 `.env.example` 为 `.env`，填写 DashScope Key 和本地服务配置。

```powershell
Copy-Item .env.example .env
```

旧版本中出现过真实 DashScope Key，已经从源码移除；请在控制台轮换旧 Key。

## 启动数据库

```powershell
docker compose -f docker-compose.legal.yml up -d
docker compose -f docker-compose.legal.yml ps
```

本项目使用独立容器：`legal-mysql`、`legal-redis`、`legal-etcd`、`legal-minio`、`legal-milvus`，不会复用当前 Dify 的 Weaviate。

## 离线入库

```powershell
python -m legal_brain.data.corpus
python -m legal_brain.data.pipeline
python -m legal_brain.rag.vector_store
```

官方来源清单模板在 `data_sources/official_legal_sources.example.yml`。生产验收只使用 `data/legal_corpus/` 中的官方下载语料和 manifest，不使用手写种子数据作为验收依据。

## FAQ 入库

```powershell
python -m legal_brain.data.faq_importer data/faq/legal_faq.jsonl
```

FAQ 数据支持 JSONL、JSON 和 CSV。字段至少包含 `question` 和 `answer`，可选字段包括 `category`、`source`、`source_type`、`review_status`、`faq_version`、`corpus_version`。

## RAGAS 评估

```powershell
python -m legal_brain.evaluation.ragas_runner data/eval/ragas_legal_eval.jsonl --run-name legal-rag-baseline
```

评估结果写入 MySQL `ragas_eval_runs`，不伪造指标；缺少 RAGAS 依赖时直接失败。

## 启动服务

```powershell
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 10000
```

访问 `http://127.0.0.1:10000/`。

## 前端工作台

正式前端在 `frontend/`，采用 Vue 3 + Vite + TypeScript + Element Plus。

```powershell
cd frontend
pnpm install
pnpm run dev
```

访问 `http://127.0.0.1:5173/`。当前工作台覆盖法律咨询、FAQ 管理、知识库 manifest、RAGAS 评估、系统状态和 Agent 下一阶段入口。

## Git 发布策略

- 稳定分支：`main`
- 远端：`git@github.com:Chatbot-zhou/Smart-RAG.git`
- 每个可用里程碑打 `usable-*` 标签。
- 只保留最近 5 个 `usable-*` 标签。

法律回答仅供业务参考，不构成正式法律意见。
