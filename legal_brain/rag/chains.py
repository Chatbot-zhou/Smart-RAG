from collections.abc import Iterator

from legal_brain.config import settings
from legal_brain.prompts import LegalPrompts


class LegalRAGChains:
    def __init__(self, llm_client):
        try:
            from langchain_core.prompts import PromptTemplate as LangChainPromptTemplate
        except ImportError as exc:
            raise RuntimeError("LangChain 生产依赖缺失：需要 langchain-core。") from exc
        self.llm_client = llm_client
        self.query_rewrite_prompt = LangChainPromptTemplate.from_template(
            LegalPrompts.query_rewrite_prompt().template
        )
        self.answer_prompt = LangChainPromptTemplate.from_template(LegalPrompts.answer_prompt().template)

    def rewrite_query(self, original_query: str) -> str:
        prompt = self.query_rewrite_prompt.format(query=original_query)
        completion = self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "你是企业法务 RAG 检索 query 改写器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=60,
        )
        rewritten = completion.choices[0].message.content.strip() if completion.choices else original_query
        return rewritten or original_query

    def stream_answer(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        context: str,
        history: str,
    ) -> Iterator[str]:
        prompt = self.answer_prompt.format(
            context=context or "未检索到足够上下文。",
            history=history,
            question=original_query,
            disclaimer=settings.service_disclaimer,
        )
        completion = self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "你是严谨的中国大陆公司法务检索增强助手。"},
                {
                    "role": "user",
                    "content": f"检索 query：{rewritten_query}\n\n{prompt}",
                },
            ],
            stream=True,
            timeout=60,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
