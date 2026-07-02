import json
from pathlib import Path
from typing import Any

from langchain.docstore.document import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from sentence_transformers import CrossEncoder

from legal_brain.config import settings
from legal_brain.logging import logger


def _device() -> str:
    return "cpu" if settings.model_device == "auto" else settings.model_device


def sparse_to_dict(row: Any) -> dict:
    indices = getattr(row, "col", None)
    if indices is None:
        indices = getattr(row, "indices", [])
    values = getattr(row, "data", [])
    return {index: value for index, value in zip(indices, values)}


class LegalVectorStore:
    def __init__(self):
        self.device = _device()
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=settings.embedding_model,
            use_f16=self.device != "cpu",
            device=self.device,
        )
        self.reranker = CrossEncoder(settings.reranker_model, device=self.device)
        self.dense_dim = self.embedding_function.dim["dense"]
        self.client = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            db_name=settings.milvus_database_name,
        )
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.has_collection(settings.milvus_collection_name):
            self.client.load_collection(settings.milvus_collection_name)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("document_id", DataType.VARCHAR, max_length=80)
        schema.add_field("title", DataType.VARCHAR, max_length=255)
        schema.add_field("domain", DataType.VARCHAR, max_length=80)
        schema.add_field("authority", DataType.VARCHAR, max_length=255)
        schema.add_field("source_url", DataType.VARCHAR, max_length=1000)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=120)
        schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        self.client.create_collection(settings.milvus_collection_name, schema=schema, index_params=index_params)
        self.client.load_collection(settings.milvus_collection_name)
        logger.info(f"已创建法律知识库集合：{settings.milvus_collection_name}")

    def upsert_chunks_from_jsonl(self, chunk_file: Path | None = None) -> int:
        chunk_file = chunk_file or settings.chunk_output_file
        chunks = [json.loads(line) for line in Path(chunk_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            return 0
        embeddings = self.embedding_function(texts)
        payload = []
        for index, chunk in enumerate(chunks):
            payload.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "dense_vector": embeddings["dense"][index],
                    "sparse_vector": sparse_to_dict(embeddings["sparse"][index]),
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "domain": chunk["domain"],
                    "authority": chunk["authority"],
                    "source_url": chunk["source_url"],
                    "parent_id": chunk["parent_id"],
                    "parent_content": chunk["parent_content"],
                }
            )
        self.client.upsert(collection_name=settings.milvus_collection_name, data=payload)
        logger.info(f"法律切块写入 Milvus：{len(payload)}")
        return len(payload)

    def search(self, query: str, domain: str | None = None) -> list[Document]:
        query_embeddings = self.embedding_function([query])
        filter_expr = f'domain == "{domain}"' if domain else ""
        dense_request = AnnSearchRequest(
            data=[query_embeddings["dense"][0]],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=settings.retrieval_k,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_to_dict(query_embeddings["sparse"][0])],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=settings.retrieval_k,
            expr=filter_expr,
        )
        hits = self.client.hybrid_search(
            collection_name=settings.milvus_collection_name,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(0.7, 1.0),
            limit=settings.retrieval_k,
            output_fields=["text", "document_id", "title", "domain", "authority", "source_url", "parent_id", "parent_content"],
        )[0]
        parent_docs = self._unique_parent_docs([hit["entity"] for hit in hits])
        if len(parent_docs) < 2:
            return parent_docs
        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs)
        return [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)][: settings.candidate_m]

    def _unique_parent_docs(self, hits: list[dict[str, Any]]) -> list[Document]:
        seen = set()
        docs = []
        for hit in hits:
            parent_content = hit.get("parent_content") or hit["text"]
            if parent_content in seen:
                continue
            seen.add(parent_content)
            docs.append(
                Document(
                    page_content=parent_content,
                    metadata={
                        "document_id": hit.get("document_id"),
                        "title": hit.get("title"),
                        "domain": hit.get("domain"),
                        "authority": hit.get("authority"),
                        "source_url": hit.get("source_url"),
                        "parent_id": hit.get("parent_id"),
                    },
                )
            )
        return docs


if __name__ == "__main__":
    LegalVectorStore().upsert_chunks_from_jsonl()

