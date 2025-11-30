from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime, timezone



class Message(BaseModel):
    message_id: UUID = Field(default_factory=uuid4)
    role: Literal["system", "user", "assistant", "tool", "agent"]  # extensible
    content: str
    name: Optional[str] = None  # for agent names or system labels
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Optional metadata
    tool_calls: Optional[Dict[str, Any]] = Field(default_factory=dict)
    logs: Optional[List[str]] = Field(default_factory=list)
    trace_id: Optional[str] = None
    parent_id: Optional[UUID] = None
    usage: Optional[Dict[str, Any]] = Field(default_factory=dict)
    latency_ms: Optional[float] = None

