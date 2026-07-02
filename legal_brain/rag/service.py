from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from legal_brain.config import settings
from legal_brain.intent import intent_recognizer
from legal_brain.logging import logger
from legal_brain.rag.chains import LegalRAGChains
from legal_brain.strategy import LegalRetrievalStrategySelector


@dataclass
class PreparedRAGAnswer:
    original_query: str
    rewritten_query: str
    references: list[dict[str, Any]]
    stream: Iterator[str]


class LegalRAGService:
    def __init__(self):
        settings.require_llm()
        try:
            from openai import OpenAI
            from legal_brain.rag.vector_store import LegalVectorStore
        except ImportError as exc:
            raise RuntimeError("法律问答依赖未安装，请先安装 requirements.txt。") from exc

        self.client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
        self.vector_store = LegalVectorStore()
        self.strategy_selector = LegalRetrievalStrategySelector()
        self.chains = LegalRAGChains(self.client)

    def answer(self, query: str, source_filter: str | None = None, history: list[dict] | None = None) -> Iterator[str]:
        prepared = self.prepare_answer(query, source_filter=source_filter, history=history)
        yield from prepared.stream

    def prepare_answer(
        self, query: str, source_filter: str | None = None, history: list[dict] | None = None
    ) -> PreparedRAGAnswer:
        intent, confidence = intent_recognizer.classify(query)
        logger.info(f"法律意图识别：intent={intent}, confidence={confidence:.2f}")
        strategy = self.strategy_selector.select_strategy(query)
        logger.info(f"法律检索策略：{strategy}")
        rewritten_query = self.chains.rewrite_query(query)
        docs = self.vector_store.search(rewritten_query, domain=source_filter)
        context = "\n\n".join(
            [
                f"来源：《{doc.metadata.get('title')}》 {doc.metadata.get('authority')} {doc.metadata.get('source_url')}\n{doc.page_content}"
                for doc in docs
            ]
        )
        history_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in (history or [])[-5:]])
        references = [doc.metadata for doc in docs]
        return PreparedRAGAnswer(
            original_query=query,
            rewritten_query=rewritten_query,
            references=references,
            stream=self.chains.stream_answer(
                original_query=query,
                rewritten_query=rewritten_query,
                context=context,
                history=history_text,
            ),
        )

    def retrieve_references(self, query: str, source_filter: str | None = None) -> list[dict]:
        docs = self.vector_store.search(query, domain=source_filter)
        return [doc.metadata for doc in docs]
