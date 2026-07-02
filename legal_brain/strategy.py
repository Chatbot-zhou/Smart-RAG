from openai import OpenAI

from legal_brain.config import settings
from legal_brain.logging import logger
from legal_brain.prompts import LegalPrompts


class LegalRetrievalStrategySelector:
    VALID_STRATEGIES = {"直接检索", "假设问题检索", "子查询检索", "回溯问题检索"}

    def __init__(self):
        self.prompt = LegalPrompts.strategy_prompt()
        self.client: OpenAI | None = None

    def _client(self) -> OpenAI:
        settings.require_llm()
        if self.client is None:
            self.client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
        return self.client

    def select_strategy(self, query: str) -> str:
        try:
            completion = self._client().chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "你严格按要求选择法律检索策略。"},
                    {"role": "user", "content": self.prompt.format(query=query)},
                ],
                temperature=0.1,
            )
            strategy = completion.choices[0].message.content.strip() if completion.choices else "直接检索"
            return strategy if strategy in self.VALID_STRATEGIES else "直接检索"
        except Exception as exc:
            logger.warning(f"策略选择失败，回退到直接检索: {exc}")
            return "直接检索"

