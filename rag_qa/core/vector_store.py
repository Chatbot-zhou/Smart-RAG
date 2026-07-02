import hashlib

from langchain.docstore.document import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from sentence_transformers import CrossEncoder

from base.config import single_config as config
from base.logger import single_logger as logger


def _model_device() -> str:
    return "cpu" if config.MODEL_DEVICE == "auto" else config.MODEL_DEVICE


def _sparse_vector_to_dict(row) -> dict:
    indices = getattr(row, "col", None)
    if indices is None:
        indices = getattr(row, "indices", [])
    values = getattr(row, "data", [])
    return {token_id: value for token_id, value in zip(indices, values)}


class VectorStore:
    def __init__(
        self,
        collection_name=config.MILVUS_COLLECTION_NAME,
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
        database=config.MILVUS_DATABASE_NAME,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.device = _model_device()

        self.reranker = CrossEncoder(config.RERANKER_MODEL, device=self.device)
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=config.EMBEDDING_MODEL,
            use_f16=self.device != "cpu",
            device=self.device,
        )
        self.dense_dim = self.embedding_function.dim["dense"]
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

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
            self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
            logger.info(f"已创建集合 {self.collection_name}")
        else:
            logger.info(f"集合已存在 {self.collection_name}")

        self.client.load_collection(self.collection_name)
        logger.info(f"已加载集合 {self.collection_name}")

    def add_documents(self, documents: list[Document]):
        texts = [document.page_content for document in documents]
        if not texts:
            logger.warning("文档数据为空")
            return

        embeddings = self.embedding_function(texts)
        data = []
        for index, doc in enumerate(documents):
            content = doc.page_content.encode("utf-8")
            data.append(
                {
                    "id": hashlib.md5(content).hexdigest(),
                    "text": doc.page_content,
                    "dense_vector": embeddings["dense"][index],
                    "sparse_vector": _sparse_vector_to_dict(embeddings["sparse"][index]),
                    "parent_id": doc.metadata["parent_id"],
                    "parent_content": doc.metadata["parent_content"],
                    "source": doc.metadata.get("source", "unknown"),
                    "timestamp": doc.metadata.get("timestamp", "unknown"),
                }
            )

        self.client.upsert(collection_name=self.collection_name, data=data)
        logger.info(f"成功写入 Milvus：{len(data)} 条")

    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, source_filter=None):
        query_embeddings = self.embedding_function([query])
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = _sparse_vector_to_dict(query_embeddings["sparse"][0])
        filter_expr = f'source == "{source_filter}"' if source_filter else ""

        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(0.7, 1.0),
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"],
        )[0]

        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        if len(parent_docs) < 2:
            return parent_docs

        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs)
        ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        return ranked_parent_docs[: config.CANDIDATE_M]

    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit["text"],
            metadata={
                "source": hit["source"],
                "timestamp": hit["timestamp"],
                "parent_id": hit["parent_id"],
                "parent_content": hit["parent_content"],
            },
        )

    def _get_unique_parent_docs(self, sub_chunks):
        parent_contents = set()
        unique_parent_docs = []

        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                unique_parent_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)

        return unique_parent_docs


if __name__ == "__main__":
    import rag_qa.core.document_processor as document_processor

    documents = document_processor.process_documents(config.DOCUMENT_DATA_DIR)
    VectorStore().add_documents(documents)
