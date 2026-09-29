from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryObject:
    memory_id: str
    conversation_id: str
    source_event_id: str
    text: str
    memory_type: str = "event"
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferenceMemory:
    memory_id: str
    conversation_id: str
    content: str
    preference_type: str = "preference"
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionMemory:
    memory_id: str
    conversation_id: str
    tool_name: str
    action: str
    result_summary: str
    status: str = "unknown"
    evidence_ref: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeMemory:
    episode_id: str
    conversation_id: str
    memory_ids: list[str]
    summary: str
    start_time: str | None = None
    end_time: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryMemory:
    trajectory_id: str
    conversation_id: str
    episode_ids: list[str]
    title: str
    summary: str
    relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
