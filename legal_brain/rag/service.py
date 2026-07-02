from collections.abc import Iterator

from openai import OpenAI

from legal_brain.config import settings
from legal_brain.intent import intent_recognizer
from legal_brain.logging import logger
from legal_brain.prompts import LegalPrompts
from legal_brain.rag.vector_store import LegalVectorStore
from legal_brain.strategy import LegalRetrievalStrategySelector


class LegalRAGService:
    def __init__(self):
        settings.require_llm()
        self.client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
        self.vector_store = LegalVectorStore()
        self.strategy_selector = LegalRetrievalStrategySelector()
        self.answer_prompt = LegalPrompts.answer_prompt()

    def call_llm(self, prompt: str) -> Iterator[str]:
        completion = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "你是严谨的中国大陆公司法务检索增强助手。"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            timeout=60,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def answer(self, query: str, source_filter: str | None = None, history: list[dict] | None = None) -> Iterator[str]:
        intent, confidence = intent_recognizer.classify(query)
        logger.info(f"法律意图识别：intent={intent}, confidence={confidence:.2f}")
        strategy = self.strategy_selector.select_strategy(query)
        logger.info(f"法律检索策略：{strategy}")
        docs = self.vector_store.search(query, domain=source_filter)
        context = "\n\n".join(
            [
                f"来源：《{doc.metadata.get('title')}》 {doc.metadata.get('authority')} {doc.metadata.get('source_url')}\n{doc.page_content}"
                for doc in docs
            ]
        )
        history_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in (history or [])[-5:]])
        prompt = self.answer_prompt.format(
            context=context or "未检索到足够上下文。",
            history=history_text,
            question=query,
            disclaimer=settings.service_disclaimer,
        )
        yield from self.call_llm(prompt)

    def retrieve_references(self, query: str, source_filter: str | None = None) -> list[dict]:
        docs = self.vector_store.search(query, domain=source_filter)
        return [doc.metadata for doc in docs]

