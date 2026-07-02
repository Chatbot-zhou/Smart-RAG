from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class A2AMessage:
    sender: str
    receiver: str
    task: str
    payload: dict[str, Any]
    trace_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

