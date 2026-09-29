from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_EVENT = "system_event"
    WORKSPACE_ACTION = "workspace_action"
    TOOL_CALL = "tool_call"
    TOOL_OBSERVATION = "tool_observation"
    TOOL_OBSERVATION_SUMMARY = "tool_observation_summary"
    EXECUTION_SUMMARY = "execution_summary"


@dataclass(slots=True)
class CogMemEvent:
    event_type: str
    text: str = ""
    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    speaker: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    summary: str | None = None
    status: str | None = None
    store_policy: str = "memory_summary"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CogMemEvent":
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        if not event_type:
            raise ValueError("event_type is required")
        value = payload.get("arguments") or payload.get("args") or {}
        return cls(
            event_type=event_type,
            text=str(payload.get("text") or payload.get("content") or ""),
            event_id=str(payload.get("event_id") or payload.get("id") or f"evt_{uuid4().hex}"),
            timestamp=str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            speaker=optional_string(payload.get("speaker")),
            tool_name=optional_string(payload.get("tool_name") or payload.get("tool")),
            arguments=dict(value) if isinstance(value, dict) else None,
            result=payload.get("result"),
            summary=optional_string(payload.get("summary")),
            status=optional_string(payload.get("status")),
            store_policy=str(payload.get("store_policy") or "memory_summary"),
            metadata=dict(payload.get("metadata") or {}),
        )


def normalize_events(events: list[dict[str, Any] | CogMemEvent]) -> list[CogMemEvent]:
    normalized: list[CogMemEvent] = []
    for event in events:
        if isinstance(event, CogMemEvent):
            normalized.append(event)
        elif isinstance(event, dict):
            normalized.append(CogMemEvent.from_dict(event))
        else:
            raise TypeError(f"Unsupported event payload: {type(event)!r}")
    return normalized


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
