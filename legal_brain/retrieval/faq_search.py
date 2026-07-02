from dataclasses import dataclass
from typing import Any

import numpy as np

from legal_brain.config import settings
from legal_brain.retrieval.preprocess import normalize_query
from legal_brain.storage.mysql import LegalMySQLStore


@dataclass
class FAQMatch:
    item: dict[str, Any]
    bm25_score: float
    embedding_score: float
    final_score: float


class FAQSearchService:
    def __init__(self, store: LegalMySQLStore):
        try:
            import jieba
            from milvus_model.hybrid import BGEM3EmbeddingFunction
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("FAQ 检索生产依赖缺失：需要 jieba、rank-bm25、milvus-model 和 BGE-M3。") from exc

        self.store = store
        self.jieba = jieba
        self.BM25Okapi = BM25Okapi
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=settings.embedding_model,
            use_f16=settings.model_device not in ("auto", "cpu"),
            device="cpu" if settings.model_device == "auto" else settings.model_device,
        )
        self.items: list[dict[str, Any]] = []
        self.bm25 = None
        self.question_embeddings: np.ndarray | None = None
        self.refresh()

    def refresh(self) -> None:
        self.items = self.store.list_active_faq_items(corpus_version=settings.corpus_version)
        tokenized = [self._tokenize(item["normalized_question"]) for item in self.items]
        self.bm25 = self.BM25Okapi(tokenized) if tokenized else None
        if self.items:
            embeddings = self.embedding_function([item["normalized_question"] for item in self.items])
            self.question_embeddings = np.array(embeddings["dense"], dtype=np.float32)
        else:
            self.question_embeddings = None

    def search(self, query: str) -> FAQMatch | None:
        if not self.items or self.bm25 is None or self.question_embeddings is None:
            return None
        normalized = normalize_query(query)
        bm25_scores = np.array(self.bm25.get_scores(self._tokenize(normalized)), dtype=np.float32)
        bm25_scores = self._normalize_scores(bm25_scores)
        query_embedding = np.array(self.embedding_function([normalized])["dense"][0], dtype=np.float32)
        embedding_scores = self._cosine_scores(query_embedding, self.question_embeddings)
        final_scores = settings.faq_embedding_weight * embedding_scores + settings.faq_bm25_weight * bm25_scores
        best_index = int(np.argmax(final_scores))
        return FAQMatch(
            item=self.items[best_index],
            bm25_score=float(bm25_scores[best_index]),
            embedding_score=float(embedding_scores[best_index]),
            final_score=float(final_scores[best_index]),
        )

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in self.jieba.lcut(text) if token.strip()]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        max_score = float(scores.max())
        if max_score <= 0:
            return np.zeros_like(scores)
        return scores / max_score

    def _cosine_scores(self, query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_embedding)
        matrix_norm = np.linalg.norm(matrix, axis=1)
        denom = np.maximum(query_norm * matrix_norm, 1e-12)
        return np.dot(matrix, query_embedding) / denom
