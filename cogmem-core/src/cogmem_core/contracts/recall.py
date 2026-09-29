from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RecallRequest:
    conversation_id: str
    query: str
    include_temporal: bool = True
    include_preferences: bool = True
    include_execution: bool = False
    include_superseded: bool = False
    limit: int = 8
    token_budget: int | None = None


@dataclass(slots=True)
class ContextMemoryItem:
    item_id: str
    text: str
    source: str
    score: float = 0.0
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CognitiveContext:
    query: str
    conversation_id: str
    preference_memory: list[ContextMemoryItem] = field(default_factory=list)
    long_term_background: list[ContextMemoryItem] = field(default_factory=list)
    current_situation: list[ContextMemoryItem] = field(default_factory=list)
    related_experience: list[ContextMemoryItem] = field(default_factory=list)
    supporting_evidence: list[ContextMemoryItem] = field(default_factory=list)
    temporal_memory: list[ContextMemoryItem] = field(default_factory=list)
    execution_memory: list[ContextMemoryItem] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "conversation_id": self.conversation_id,
            "preference_memory": [item.to_dict() for item in self.preference_memory],
            "long_term_background": [item.to_dict() for item in self.long_term_background],
            "current_situation": [item.to_dict() for item in self.current_situation],
            "related_experience": [item.to_dict() for item in self.related_experience],
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "temporal_memory": [item.to_dict() for item in self.temporal_memory],
            "execution_memory": [item.to_dict() for item in self.execution_memory],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(slots=True)
class ContextPack:
    conversation_id: str
    query: str
    cognitive_context: CognitiveContext
    prompt_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "query": self.query,
            "cognitive_context": self.cognitive_context.to_dict(),
            "prompt_text": self.prompt_text,
        }
