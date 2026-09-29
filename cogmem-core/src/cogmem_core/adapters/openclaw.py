from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from ..contracts.events import CogMemEvent, EventType
from ..contracts.recall import ContextPack

MAX_TEXT_CHARS = 800
MAX_ARGUMENT_CHARS = 240
MAX_OBSERVATION_CHARS = 400
CONTROL_RECORD_TYPES = {
    "session",
    "model_change",
    "thinking_level_change",
    "custom",
}
HOUSEKEEPING_MARKERS = (
    "pre-compaction memory flush",
    "if nothing to store, reply with no_reply",
)
UNTRUSTED_BLOCK = re.compile(
    r"(?:Conversation info|Sender)\s+\(untrusted metadata\):\s*```json\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b[A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|AUTHORIZATION)\b\s*[=:]\s*)([^\s\"']+)"
)
SECRET_TOKEN = re.compile(r"\b(?:tvly|sk|ghp|gho|xoxb|xoxp)-[A-Za-z0-9_-]{8,}\b")


@dataclass(slots=True)
class OpenClawConversion:
    conversation_id: str
    events: list[CogMemEvent]
    skipped: int = 0
    record_count: int = 0
    workspace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenClawAdapter:
    """Translate OpenClaw session logs and live payloads into CogMem events."""

    def to_events(self, payload: Any) -> list[CogMemEvent]:
        return self.convert(payload).events

    def convert(self, payload: Any) -> OpenClawConversion:
        records = list(iter_records(payload))
        conversation_id = ""
        workspace: str | None = None
        events: list[CogMemEvent] = []
        skipped = 0
        last_assistant_text: str | None = None
        default_speaker: str | None = None

        for record in records:
            if not isinstance(record, dict):
                skipped += 1
                continue

            record_type = str(record.get("type") or "")
            if record_type == "session":
                conversation_id = conversation_id or str(record.get("id") or "")
                workspace = optional_string(record.get("cwd")) or workspace
                continue
            if record_type in CONTROL_RECORD_TYPES:
                nested = record.get("data") if isinstance(record.get("data"), dict) else {}
                conversation_id = conversation_id or str(nested.get("sessionId") or "")
                continue

            produced = convert_message_record(
                record,
                default_speaker=default_speaker,
                last_assistant_text=last_assistant_text,
            )
            if not produced:
                skipped += 1
                continue

            events.extend(produced)
            for event in produced:
                if event.event_type == EventType.USER_MESSAGE.value and event.speaker:
                    default_speaker = event.speaker
                if event.event_type == EventType.ASSISTANT_MESSAGE.value and event.text:
                    last_assistant_text = event.text

        conversation_id = conversation_id or conversation_id_from_payload(payload) or "openclaw_unknown"
        return OpenClawConversion(
            conversation_id=conversation_id,
            events=events,
            skipped=skipped,
            record_count=len(records),
            workspace=workspace,
            metadata={"source": "openclaw", "workspace": workspace},
        )

    def from_context_pack(self, context: ContextPack) -> dict[str, Any]:
        return {
            "conversationId": context.conversation_id,
            "prependContext": context.prompt_text,
            "source": "cogmem",
        }


def iter_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, (str, Path)):
        path = Path(payload)
        if path.exists() and path.is_file():
            yield from read_jsonl(path)
            return
        text = str(payload).strip()
        if "\n" in text or text.startswith("{"):
            yield from parse_jsonl_text(text)
            return
        raise FileNotFoundError(f"OpenClaw payload path does not exist: {payload}")

    if isinstance(payload, dict):
        for key in ("records", "messages", "events", "log", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, str):
                yield from parse_jsonl_text(item)
        return

    raise TypeError(f"Unsupported OpenClaw payload: {type(payload)!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return parse_jsonl_text(path.read_text(encoding="utf-8"))


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stripped = text.strip()
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def conversation_id_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("conversation_id", "conversationId", "session_id", "sessionId", "id"):
            value = payload.get(key)
            if value:
                return str(value)
    return ""


def convert_message_record(
    record: dict[str, Any],
    *,
    default_speaker: str | None,
    last_assistant_text: str | None,
) -> list[CogMemEvent]:
    message = record.get("message") if isinstance(record.get("message"), dict) else record
    role = str(message.get("role") or "")
    timestamp = str(
        record.get("timestamp")
        or message.get("timestamp")
        or ""
    )
    record_id = str(record.get("id") or message.get("id") or "")
    content = message.get("content")

    if role == "user":
        return convert_user_message(
            content,
            event_id=record_id,
            timestamp=timestamp,
            default_speaker=default_speaker,
        )
    if role == "assistant":
        return convert_assistant_message(
            message,
            event_id=record_id,
            timestamp=timestamp,
            last_assistant_text=last_assistant_text,
        )
    if role == "toolResult" or message.get("toolCallId") or message.get("toolName"):
        event = convert_tool_result(message, event_id=record_id, timestamp=timestamp)
        return [event] if event else []
    return []


def convert_user_message(
    content: Any,
    *,
    event_id: str,
    timestamp: str,
    default_speaker: str | None,
) -> list[CogMemEvent]:
    raw_text, media_notes = collect_text_parts(content)
    user_text, extracted = unwrap_openclaw_user_text(raw_text)
    if is_housekeeping_text(user_text):
        return []
    if not user_text and not media_notes:
        return []

    speaker = (
        optional_string(extracted.get("sender"))
        or optional_string(extracted.get("name"))
        or default_speaker
    )
    text = user_text or "; ".join(media_notes)
    metadata = {
        "platform": "openclaw",
        "sender_id": extracted.get("sender_id") or extracted.get("id"),
        "channel_message_id": extracted.get("message_id"),
    }
    if media_notes:
        metadata["media"] = media_notes
    return [
        CogMemEvent(
            event_type=EventType.USER_MESSAGE.value,
            text=text,
            event_id=event_id or default_event_id("user"),
            timestamp=timestamp or now_iso(),
            speaker=speaker,
            metadata={key: value for key, value in metadata.items() if value},
        )
    ]


def convert_assistant_message(
    message: dict[str, Any],
    *,
    event_id: str,
    timestamp: str,
    last_assistant_text: str | None,
) -> list[CogMemEvent]:
    stop_reason = str(message.get("stopReason") or "")
    content = message.get("content")
    text, _media_notes = collect_text_parts(content, include_tools=False)
    text = text.strip()
    events: list[CogMemEvent] = []

    if text and text != "NO_REPLY" and text != last_assistant_text:
        events.append(
            CogMemEvent(
                event_type=EventType.ASSISTANT_MESSAGE.value,
                text=truncate(text, MAX_TEXT_CHARS),
                event_id=event_id or default_event_id("assistant"),
                timestamp=timestamp or now_iso(),
                speaker="assistant",
                status=stop_reason or None,
                metadata={
                    key: value
                    for key, value in {
                        "platform": "openclaw",
                        "model": message.get("model"),
                        "provider": message.get("provider"),
                    }.items()
                    if value
                },
            )
        )
    elif not text and stop_reason in {"error", "aborted"}:
        return []

    for index, part in enumerate(iter_content_parts(content)):
        if str(part.get("type") or "") != "toolCall":
            continue
        events.append(convert_tool_call(part, parent_id=event_id, timestamp=timestamp, index=index))
    return events


def convert_tool_call(
    part: dict[str, Any],
    *,
    parent_id: str,
    timestamp: str,
    index: int,
) -> CogMemEvent:
    tool_name = optional_string(part.get("name") or part.get("toolName")) or "unknown_tool"
    raw_arguments = part.get("arguments") if isinstance(part.get("arguments"), dict) else {}
    arguments = compact_arguments(raw_arguments)
    event_id = str(part.get("id") or f"{parent_id}:tool:{index}" or default_event_id("tool"))
    return CogMemEvent(
        event_type=EventType.TOOL_CALL.value,
        text=summarize_tool_call(tool_name, arguments),
        event_id=event_id,
        timestamp=timestamp or now_iso(),
        tool_name=tool_name,
        arguments=arguments,
        store_policy="memory_summary",
        metadata={"platform": "openclaw", "parent_id": parent_id},
    )


def convert_tool_result(
    message: dict[str, Any],
    *,
    event_id: str,
    timestamp: str,
) -> CogMemEvent | None:
    tool_name = optional_string(message.get("toolName") or message.get("tool_name")) or "unknown_tool"
    is_error = bool(message.get("isError"))
    details = message.get("details") if isinstance(message.get("details"), dict) else {}
    text, media_notes = collect_text_parts(message.get("content"), skip_image_data=True)
    if not text and media_notes:
        text = "; ".join(media_notes)
    if not text and details:
        text = json.dumps(details, ensure_ascii=False)
    text = redact_secrets(text)
    if not text:
        return None
    status = "error" if is_error else optional_string(details.get("status")) or "success"
    summary = summarize_tool_observation(tool_name, text, status)
    return CogMemEvent(
        event_type=EventType.TOOL_OBSERVATION.value,
        text=summary,
        event_id=event_id or str(message.get("toolCallId") or default_event_id("tool_result")),
        timestamp=timestamp or now_iso(),
        tool_name=tool_name,
        result=truncate(text, MAX_OBSERVATION_CHARS),
        summary=summary,
        status=status,
        store_policy="memory_summary",
        metadata={
            key: value
            for key, value in {
                "platform": "openclaw",
                "tool_call_id": message.get("toolCallId"),
                "media": media_notes or None,
            }.items()
            if value
        },
    )


def unwrap_openclaw_user_text(raw_text: str) -> tuple[str, dict[str, Any]]:
    extracted: dict[str, Any] = {}
    last_end = 0
    for match in UNTRUSTED_BLOCK.finditer(raw_text):
        last_end = match.end()
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            extracted.update(parsed)
    remainder = raw_text[last_end:].strip() if last_end else raw_text.strip()
    remainder = strip_system_prefix(remainder)
    return remainder.strip(), extracted


def strip_system_prefix(text: str) -> str:
    if not text.startswith("System:"):
        return text
    lines = text.splitlines()
    kept: list[str] = []
    seen_user_block = False
    for line in lines:
        if line.startswith("System:"):
            if seen_user_block:
                kept.append(line)
            continue
        if line.strip():
            seen_user_block = True
            kept.append(line)
        elif seen_user_block:
            kept.append(line)
    return "\n".join(kept).strip()


def is_housekeeping_text(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered or lowered == "no_reply":
        return True
    return any(marker in lowered for marker in HOUSEKEEPING_MARKERS)


def collect_text_parts(
    content: Any,
    *,
    include_tools: bool = False,
    skip_image_data: bool = True,
) -> tuple[str, list[str]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content.strip(), []
    texts: list[str] = []
    media_notes: list[str] = []
    for part in iter_content_parts(content):
        part_type = str(part.get("type") or "text")
        if part_type == "text":
            texts.append(str(part.get("text") or ""))
        elif part_type == "image":
            media_notes.append("image attached")
            if not skip_image_data and part.get("data"):
                texts.append("[image omitted]")
        elif part_type == "toolCall" and include_tools:
            texts.append(summarize_tool_call(str(part.get("name") or "tool"), compact_arguments(part.get("arguments") or {})))
    return "\n".join(part for part in texts if part).strip(), media_notes


def iter_content_parts(content: Any) -> Iterable[dict[str, Any]]:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                yield part
            elif isinstance(part, str):
                yield {"type": "text", "text": part}
    elif isinstance(content, dict):
        yield content
    elif isinstance(content, str):
        yield {"type": "text", "text": content}


def compact_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key, raw in value.items():
        if key in {"content", "oldText", "old_string", "new_string", "newText"} and isinstance(raw, str):
            compacted[key] = truncate(redact_secrets(raw), MAX_ARGUMENT_CHARS)
            continue
        if isinstance(raw, str):
            compacted[key] = truncate(redact_secrets(raw), MAX_ARGUMENT_CHARS)
        elif isinstance(raw, (int, float, bool)) or raw is None:
            compacted[key] = raw
        else:
            compacted[key] = truncate(redact_secrets(json.dumps(raw, ensure_ascii=False)), MAX_ARGUMENT_CHARS)
    return compacted


def summarize_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    focus_keys = ("path", "command", "query", "url", "file")
    details = [f"{key}={arguments[key]}" for key in focus_keys if arguments.get(key)]
    if not details:
        details = [f"{key}={value}" for key, value in list(arguments.items())[:2]]
    suffix = " " + " ".join(str(item) for item in details) if details else ""
    return redact_secrets(f"tool_call {tool_name}{suffix}").strip()


def summarize_tool_observation(tool_name: str, text: str, status: str) -> str:
    compact = " ".join(text.split())
    return truncate(f"tool_result {tool_name} {status}: {compact}", MAX_OBSERVATION_CHARS)


def redact_secrets(text: str) -> str:
    redacted = SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    return SECRET_TOKEN.sub("[REDACTED]", redacted)


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def default_event_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
