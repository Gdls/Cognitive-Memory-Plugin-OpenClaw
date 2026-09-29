from __future__ import annotations

from hashlib import sha1

from .events import CogMemEvent
from .schemas import ExecutionMemory


def extract_execution_memory(conversation_id: str, events: list[CogMemEvent]) -> list[ExecutionMemory]:
    items: list[ExecutionMemory] = []
    for event in events:
        if event.event_type not in {"tool_observation_summary", "execution_summary"}:
            continue
        text = event.summary or event.text
        if not text:
            continue
        digest = sha1(f"{conversation_id}:{event.event_id}:{text}".encode("utf-8")).hexdigest()[:12]
        items.append(
            ExecutionMemory(
                memory_id=f"exec_{digest}",
                conversation_id=conversation_id,
                tool_name=event.tool_name or "unknown_tool",
                action=event.event_type,
                result_summary=text,
                status=event.status or "unknown",
                timestamp=event.timestamp,
                metadata={"source_event_id": event.event_id},
            )
        )
    return items
