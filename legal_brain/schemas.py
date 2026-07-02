from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    source_filter: str | None = None
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float
    references: list[dict[str, Any]] = []
    disclaimer: str


class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class IntentResponse(BaseModel):
    intent: Literal["legal_qa", "regulation_query", "risk_identification", "contract_review", "contract_drafting"]
    confidence: float
    route: str


class ContractReviewRequest(BaseModel):
    contract_text: str = Field(..., min_length=1)
    contract_type: str = "general_commercial"
    review_focus: list[str] = []


class ContractDraftRequest(BaseModel):
    template_type: str | None = None
    contract_type: str | None = None
    slots: dict[str, Any]
    requirements: str | None = None

    @model_validator(mode="after")
    def ensure_template_type(self):
        if not self.template_type:
            self.template_type = self.contract_type
        if not self.template_type:
            raise ValueError("template_type or contract_type is required")
        return self


class AgentPlaceholderResponse(BaseModel):
    status: Literal["planned", "accepted"]
    intent: Literal["contract_review", "contract_drafting"]
    message: str
    required_next_steps: list[str]
    rag_tool_available: bool


class FAQImportRequest(BaseModel):
    dataset_path: str | None = None
    items: list[dict[str, Any]] = []
