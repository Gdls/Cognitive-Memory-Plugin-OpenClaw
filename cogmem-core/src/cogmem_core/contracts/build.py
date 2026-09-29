from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class BuildRequest:
    conversation_id: str
    force_rebuild: bool = False


@dataclass(slots=True)
class BuildCheckpoint:
    conversation_id: str
    last_event_id: str | None = None
    last_event_time: str | None = None
    build_version: str = "v0.1"


@dataclass(slots=True)
class BuildResult:
    conversation_id: str
    processed_events: int
    new_memories: int
    total_memories: int
    episodes: int
    trajectories: int
    preferences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
