# 智慧法务大脑

智慧法务大脑是面向中国大陆企业法务场景的法律 RAG 与合同 Agent 原型。当前版本覆盖法律问答、官方法律数据离线处理、独立向量数据库栈，以及合同审查/合同制作的 Agent 接口骨架。

## 能力

- 法律 RAG：支持民法、商法、公司法、劳动用工、合同和争议解决等场景。
- 离线入库：官方法律文本规范化、父块/子块切分、JSONL 产物、MySQL 元数据落库、Milvus 向量入库。
- 在线检索：BGE-M3 稠密+稀疏向量混合检索，CrossEncoder 重排序，输出带法律依据和免责声明。
- Agent 骨架：预留合同审查、合同制作、A2A 消息和 MCP 工具适配接口。

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
python -m legal_brain.data.pipeline
python -m legal_brain.rag.vector_store
```

首批种子数据在 `data_sources/legal_seed.yml`。默认只使用官方公开来源元数据，正式采集时应从国家法律法规数据库、最高人民法院等官方站点扩展。

## 启动服务

```powershell
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 10000
```

访问 `http://127.0.0.1:10000/`。

## Git 发布策略

- 稳定分支：`main`
- 远端：`git@github.com:Chatbot-zhou/Smart-RAG.git`
- 每个可用里程碑打 `usable-*` 标签。
- 只保留最近 5 个 `usable-*` 标签。

法律回答仅供业务参考，不构成正式法律意见。
