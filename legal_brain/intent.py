from legal_brain.config import settings


class LegalIntentRecognizer:
    """Rule-first recognizer with a future BERT classifier hook."""

    REVIEW_KEYWORDS = ("审查", "审核", "风险", "条款问题", "违约责任", "合同风险", "合规")
    DRAFT_KEYWORDS = ("起草", "生成合同", "制作合同", "合同模板", "拟一份", "填槽")

    def classify(self, text: str) -> tuple[str, float]:
        normalized = text.strip().lower()
        if any(keyword in normalized for keyword in self.REVIEW_KEYWORDS):
            return "contract_review", 0.82
        if any(keyword in normalized for keyword in self.DRAFT_KEYWORDS):
            return "contract_drafting", 0.82
        return "legal_qa", 0.7


intent_recognizer = LegalIntentRecognizer()

