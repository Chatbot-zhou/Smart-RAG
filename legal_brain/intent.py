from legal_brain.config import settings


class LegalIntentRecognizer:
    """BERT-first recognizer with a rule fallback for untrained local environments."""

    REVIEW_KEYWORDS = ("审查", "审核", "风险", "条款问题", "违约责任", "合同风险", "合规")
    DRAFT_KEYWORDS = ("起草", "生成合同", "制作合同", "合同模板", "拟一份", "填槽")
    REGULATION_KEYWORDS = ("法条", "法规", "法律依据", "第几条", "规定")
    RISK_KEYWORDS = ("风险识别", "风险等级", "法律风险", "违约风险", "合规风险")

    def __init__(self):
        self._bert_loaded = False
        self._bert_unavailable = False
        self._tokenizer = None
        self._model = None

    def classify(self, text: str) -> tuple[str, float]:
        bert_result = self._classify_with_bert(text)
        if bert_result:
            return bert_result
        normalized = text.strip().lower()
        if any(keyword in normalized for keyword in self.REVIEW_KEYWORDS):
            return "contract_review", 0.82
        if any(keyword in normalized for keyword in self.DRAFT_KEYWORDS):
            return "contract_drafting", 0.82
        if any(keyword in normalized for keyword in self.RISK_KEYWORDS):
            return "risk_identification", 0.78
        if any(keyword in normalized for keyword in self.REGULATION_KEYWORDS):
            return "regulation_query", 0.78
        return "legal_qa", 0.7

    def _classify_with_bert(self, text: str) -> tuple[str, float] | None:
        if self._bert_unavailable:
            return None
        if not self._bert_loaded:
            if not settings.query_classifier_model.exists():
                self._bert_unavailable = True
                return None
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError:
                self._bert_unavailable = True
                return None
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(str(settings.query_classifier_model))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(settings.query_classifier_model))
            self._model.eval()
            self._bert_loaded = True
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)[0]
        best_index = int(probabilities.argmax().item())
        label = settings.intent_labels[best_index] if best_index < len(settings.intent_labels) else "legal_qa"
        return label, float(probabilities[best_index].item())


intent_recognizer = LegalIntentRecognizer()
