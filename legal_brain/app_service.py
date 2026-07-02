import time
import uuid
from collections.abc import Iterator

from legal_brain.config import settings
from legal_brain.rag.service import LegalRAGService
from legal_brain.storage.mysql import LegalMySQLStore


class SmartLegalBrain:
    def __init__(self):
        self.store = LegalMySQLStore()
        self.rag: LegalRAGService | None = None

    def get_rag(self) -> LegalRAGService:
        if self.rag is None:
            self.rag = LegalRAGService()
        return self.rag

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        return self.store.recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        return self.store.clear_history(session_id)

    def query(self, query: str, source_filter: str | None = None, session_id: str | None = None) -> Iterator[tuple[str, bool]]:
        session_id = session_id or str(uuid.uuid4())
        history = self.get_session_history(session_id)
        answer = ""
        start = time.time()
        for token in self.get_rag().answer(query, source_filter=source_filter, history=history):
            answer += token
            yield token, False
        if answer:
            self.store.append_history(session_id, query, answer)
        yield "", True

    def close(self) -> None:
        self.store.close()
