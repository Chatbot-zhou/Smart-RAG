import uuid
from collections.abc import Iterator

from legal_brain.config import settings
from legal_brain.intent import intent_recognizer
from legal_brain.rag.service import LegalRAGService
from legal_brain.retrieval.preprocess import normalize_query, query_hash
from legal_brain.storage.mysql import LegalMySQLStore
from legal_brain.storage.redis_cache import LegalRedisCache


class SmartLegalBrain:
    def __init__(self):
        self.store = LegalMySQLStore()
        self.cache = LegalRedisCache()
        self.rag: LegalRAGService | None = None
        self.faq_search = None
        self.last_response_metadata: dict = {}

    def get_rag(self) -> LegalRAGService:
        if self.rag is None:
            self.rag = LegalRAGService()
        return self.rag

    def get_faq_search(self):
        if self.faq_search is None:
            from legal_brain.retrieval.faq_search import FAQSearchService

            self.faq_search = FAQSearchService(self.store)
        return self.faq_search

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        return self.store.recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        return self.store.clear_history(session_id)

    def query(self, query: str, source_filter: str | None = None, session_id: str | None = None) -> Iterator[tuple[str, bool]]:
        session_id = session_id or str(uuid.uuid4())
        intent, _ = intent_recognizer.classify(query)
        if intent not in {"legal_qa", "regulation_query", "risk_identification"}:
            answer = "当前阶段仅实现法律问答 RAG 主链路；合同审查和合同制作 Agent 将在下一阶段接入。"
            yield answer, False
            yield "", True
            return
        normalized_query = normalize_query(query)
        normalized_hash = query_hash(normalized_query)

        cached_answer = self._try_hot_cache(normalized_hash)
        if cached_answer:
            self.last_response_metadata = {
                "answer_type": cached_answer.get("answer_type", "cached"),
                "references": cached_answer.get("references", []),
                "corpus_version": cached_answer.get("corpus_version", settings.corpus_version),
                "route": "redis_mysql_hot",
            }
            self.store.append_history(session_id, query, cached_answer["answer"])
            yield cached_answer["answer"], False
            yield "", True
            return

        stored_answer = self.store.find_qa_by_normalized_query(normalized_query, settings.corpus_version)
        if stored_answer:
            hit_count = self.store.increment_qa_hit(int(stored_answer["id"]))
            self._promote_qa_if_needed(normalized_hash, stored_answer, hit_count)
            self.last_response_metadata = {
                "answer_type": stored_answer["answer_type"],
                "references": self.store.list_rag_references(int(stored_answer["id"])),
                "corpus_version": stored_answer["corpus_version"],
                "route": "mysql_qa_record",
            }
            self.store.append_history(session_id, query, stored_answer["answer"])
            yield stored_answer["answer"], False
            yield "", True
            return

        faq_match = self.get_faq_search().search(query)
        if faq_match and faq_match.final_score >= settings.faq_match_threshold:
            faq = faq_match.item
            hit_count = self.store.increment_faq_hit(int(faq["id"]))
            qa_id = self.store.insert_qa_record(
                {
                    "session_id": session_id,
                    "original_query": query,
                    "normalized_query": normalized_query,
                    "rewritten_query": "",
                    "answer": faq["answer"],
                    "answer_type": "faq",
                    "source_type": faq.get("source_type", "curated"),
                    "faq_id": faq["id"],
                    "faq_version": faq["faq_version"],
                    "corpus_version": faq["corpus_version"],
                    "disclaimer": settings.service_disclaimer,
                }
            )
            if hit_count >= settings.hot_question_promote_hits:
                self.store.mark_faq_hot(int(faq["id"]))
                self.cache.set_hot_question(
                    normalized_hash,
                    {
                        "faq_id": int(faq["id"]),
                        "qa_id": qa_id,
                        "faq_version": faq["faq_version"],
                        "corpus_version": faq["corpus_version"],
                    },
                )
            self.last_response_metadata = {
                "answer_type": "faq",
                "references": [
                    {
                        "title": faq.get("category") or "FAQ",
                        "source_url": faq.get("source"),
                        "faq_version": faq.get("faq_version"),
                    }
                ],
                "corpus_version": faq["corpus_version"],
                "route": "faq_similarity",
                "score": faq_match.final_score,
            }
            self.store.append_history(session_id, query, faq["answer"])
            yield faq["answer"], False
            yield "", True
            return

        history = self.get_session_history(session_id)
        answer = ""
        prepared = self.get_rag().prepare_answer(query, source_filter=source_filter, history=history)
        for token in prepared.stream:
            answer += token
            yield token, False
        if answer:
            self.store.append_history(session_id, query, answer)
            qa_id = self.store.insert_qa_record(
                {
                    "session_id": session_id,
                    "original_query": query,
                    "normalized_query": normalized_query,
                    "rewritten_query": prepared.rewritten_query,
                    "answer": answer,
                    "answer_type": "rag",
                    "source_type": "official_corpus",
                    "corpus_version": settings.corpus_version,
                    "disclaimer": settings.service_disclaimer,
                }
            )
            self.store.insert_rag_references(qa_id, prepared.references, settings.corpus_version)
            self.last_response_metadata = {
                "answer_type": "rag",
                "references": prepared.references,
                "corpus_version": settings.corpus_version,
                "route": "milvus_rag",
                "rewritten_query": prepared.rewritten_query,
            }
        yield "", True

    def _try_hot_cache(self, normalized_hash: str) -> dict | None:
        cached = self.cache.get_hot_question(normalized_hash)
        if not cached:
            return None
        if cached.get("corpus_version") != settings.corpus_version:
            self.cache.delete_hot_question(normalized_hash)
            return None
        if cached.get("qa_id"):
            record = self.store.get_qa_record(int(cached["qa_id"]))
            if record and record["corpus_version"] == settings.corpus_version:
                self.store.increment_qa_hit(int(record["id"]))
                self.cache.refresh_hot_question(normalized_hash, cached)
                return record
        if cached.get("faq_id"):
            faq = self.store.get_faq_item(int(cached["faq_id"]))
            if faq and faq["review_status"] == "approved" and faq["corpus_version"] == settings.corpus_version:
                self.store.increment_faq_hit(int(faq["id"]))
                self.cache.refresh_hot_question(normalized_hash, cached)
                return {"answer": faq["answer"]}
        self.cache.delete_hot_question(normalized_hash)
        return None

    def _promote_qa_if_needed(self, normalized_hash: str, record: dict, hit_count: int) -> None:
        if hit_count < settings.hot_question_promote_hits:
            return
        self.store.mark_qa_hot(int(record["id"]))
        self.cache.set_hot_question(
            normalized_hash,
            {
                "qa_id": int(record["id"]),
                "faq_id": record.get("faq_id"),
                "faq_version": record.get("faq_version"),
                "corpus_version": record["corpus_version"],
            },
        )

    def close(self) -> None:
        self.store.close()
