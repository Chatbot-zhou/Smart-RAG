from typing import Any, Literal

from pydantic import BaseModel, Field


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
    intent: Literal["legal_qa", "contract_review", "contract_drafting"]
    confidence: float
    route: str


class ContractReviewRequest(BaseModel):
    contract_text: str = Field(..., min_length=1)
    contract_type: str = "general_commercial"
    review_focus: list[str] = []


class ContractDraftRequest(BaseModel):
    template_type: str
    slots: dict[str, Any]
    requirements: str | None = None


class AgentPlaceholderResponse(BaseModel):
    status: Literal["planned", "accepted"]
    intent: Literal["contract_review", "contract_drafting"]
    message: str
    required_next_steps: list[str]
    rag_tool_available: bool

