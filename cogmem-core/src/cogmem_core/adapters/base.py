from __future__ import annotations

from typing import Any, Protocol

from ..contracts.events import CogMemEvent
from ..contracts.recall import ContextPack


class AgentPlatformAdapter(Protocol):
    """Payload translation only. Callable agent tools live in the platform plugin.

    Python adapters convert host logs into CogMemEvent and ContextPack back into
    host prompt fields. OpenClaw-style tools (search/get/timeline) are registered
    by the thin JS adapter and call the sidecar HTTP surface.
    """

    def to_events(self, payload: Any) -> list[CogMemEvent]: ...
    def from_context_pack(self, context: ContextPack) -> Any: ...
