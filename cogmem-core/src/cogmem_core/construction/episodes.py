from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..contracts.memory import EpisodeMemory
from .config import ConstructionConfig
from .llm import LlmHttpError, chat_complete, extract_json_object
from .text import compact_text, jaccard, memory_speaker, memory_text


@dataclass(slots=True)
class EpisodeSegment:
    start: int
    end: int
    title: str
    reason: str
    confidence: float
    topic: str = ""
    summary: str = ""


@dataclass(slots=True)
class RollingEpisodeConfig:
    batch_size: int = 30
    tail_keep_episode_count: int = 2


class EpisodeSegmenter(Protocol):
    def segment(self, memories: list[dict[str, Any]]) -> list[EpisodeSegment]:
        ...


class EpisodeSegmentationError(RuntimeError):
    pass


class HeuristicEpisodeSegmenter:
    """Non-LLM judge for the LoCoMo rolling window.

    Boundaries follow topic/event shifts: a new user turn with near-zero
    lexical overlap starts an episode. Replies and tool traces stay attached
    to the preceding user turn.
    """

    def __init__(self, *, split_threshold: float = 0.08) -> None:
        self.split_threshold = split_threshold

    def segment(self, memories: list[dict[str, Any]]) -> list[EpisodeSegment]:
        if not memories:
            return []
        boundaries = [0]
        for index in range(1, len(memories)):
            if self._should_split(memories, index):
                boundaries.append(index)
        boundaries.append(len(memories))
        segments: list[EpisodeSegment] = []
        for start, end in zip(boundaries, boundaries[1:]):
            chunk = memories[start:end]
            title = _title_from_memories(chunk)
            summary = compact_text(" ".join(memory_text(item) for item in chunk))[:240]
            segments.append(
                EpisodeSegment(
                    start=start,
                    end=end - 1,
                    title=title,
                    reason="heuristic topic/event boundary",
                    confidence=0.55,
                    topic=title,
                    summary=summary or title,
                )
            )
        return normalize_segments(segments, len(memories))

    def _should_split(self, memories: list[dict[str, Any]], index: int) -> bool:
        current = memories[index]
        previous = memories[index - 1]
        current_type = str((current.get("metadata") or {}).get("event_type") or "")
        previous_type = str((previous.get("metadata") or {}).get("event_type") or "")
        if current_type in {"assistant_message", "tool_call", "tool_observation", "execution_summary"}:
            return False
        if previous_type in {"tool_call"} and current_type in {"tool_observation", "tool_call"}:
            return False
        if current_type != "user_message":
            return False
        current_text = memory_text(current)
        revision_markers = ("不要", "不用", "别再", "不再", "不碰", "改成", "纠正", "其实是")
        if any(marker in current_text for marker in revision_markers):
            return True
        window = " ".join(memory_text(item) for item in memories[max(0, index - 4) : index])
        overlap = jaccard(window, current_text)
        return overlap < self.split_threshold


class OpenAICompatibleEpisodeSegmenter:
    def __init__(self, config: ConstructionConfig) -> None:
        if not config.llm_ready:
            raise ValueError("locomo_llm episode builder requires LLM credentials")
        self.config = config

    def segment(self, memories: list[dict[str, Any]]) -> list[EpisodeSegment]:
        prompt = build_episode_segmentation_prompt(memories)
        text = chat_complete(
            base_url=str(self.config.llm_base_url),
            api_key=str(self.config.llm_api_key),
            model=str(self.config.llm_model),
            prompt=prompt,
            timeout_seconds=self.config.llm_timeout_seconds,
        )
        raw_segments = parse_segments_from_text(text)
        if not raw_segments:
            raise EpisodeSegmentationError("LLM returned no parseable episode segments")
        segments = normalize_segments(raw_segments, len(memories))
        if not valid_segments(segments, len(memories)):
            raise EpisodeSegmentationError("LLM returned no valid episode segments")
        return segments


def build_rolling_episodes(
    memories: list[dict[str, Any]],
    segmenter: EpisodeSegmenter,
    *,
    sample_id: str,
    config: RollingEpisodeConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or RollingEpisodeConfig()
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if config.tail_keep_episode_count < 1:
        raise ValueError("tail_keep_episode_count must be at least 1")

    pending: list[dict[str, Any]] = []
    next_index = 0
    episodes: list[dict[str, Any]] = []

    while next_index < len(memories) or pending:
        while next_index < len(memories) and len(pending) < config.batch_size:
            pending.append(memories[next_index])
            next_index += 1
        if not pending:
            break
        is_final = next_index >= len(memories)
        segments = segmenter.segment(pending)
        if is_final:
            commit_segments = segments
        elif len(segments) > config.tail_keep_episode_count:
            commit_segments = segments[: -config.tail_keep_episode_count]
        else:
            commit_segments = segments[:1]
        if not commit_segments:
            commit_segments = [EpisodeSegment(0, 0, "fallback single memory", "no committable segment", 0.0)]

        last_committed_end = -1
        for segment in commit_segments:
            segment_memories = pending[segment.start : segment.end + 1]
            if not segment_memories:
                continue
            episode_index = len(episodes) + 1
            episodes.append(
                build_episode_payload(
                    sample_id=sample_id,
                    episode_index=episode_index,
                    segment=segment,
                    memories=segment_memories,
                )
            )
            last_committed_end = max(last_committed_end, segment.end)
        if last_committed_end < 0:
            break
        pending = pending[last_committed_end + 1 :]
    return episodes


def build_episode_payload(
    *,
    sample_id: str,
    episode_index: int,
    segment: EpisodeSegment,
    memories: list[dict[str, Any]],
) -> dict[str, Any]:
    memory_ids = [str(item.get("memory_id") or "") for item in memories if item.get("memory_id")]
    return {
        "sample_id": sample_id,
        "episode_id": f"{sample_id}_ep_{episode_index:04d}",
        "episode_index": episode_index,
        "title": segment.title,
        "topic": segment.topic or segment.title,
        "summary": segment.summary or _title_from_memories(memories),
        "reason": segment.reason,
        "confidence": segment.confidence,
        "start_time": memories[0].get("timestamp"),
        "end_time": memories[-1].get("timestamp"),
        "memory_count": len(memories),
        "memory_ids": memory_ids,
        "memories": [
            {"mem": f"mem{index}", "text": memory_text(memory), "ref": {"memory_id": memory.get("memory_id")}}
            for index, memory in enumerate(memories)
        ],
    }


def locomo_episodes_from_objects(
    conversation_id: str,
    memories: list[dict[str, Any]],
    config: ConstructionConfig,
) -> list[EpisodeMemory]:
    ordered = [item for item in memories if item.get("memory_id")]
    if not ordered:
        return []
    segmenter: EpisodeSegmenter
    builder_name = config.resolved_episode_builder()
    if builder_name == "locomo_llm":
        try:
            segmenter = OpenAICompatibleEpisodeSegmenter(config)
            payloads = build_rolling_episodes(
                ordered,
                segmenter,
                sample_id=conversation_id,
                config=RollingEpisodeConfig(
                    batch_size=config.episode_batch_size,
                    tail_keep_episode_count=config.episode_tail_keep,
                ),
            )
        except (EpisodeSegmentationError, LlmHttpError, OSError, json.JSONDecodeError):
            payloads = build_rolling_episodes(
                ordered,
                HeuristicEpisodeSegmenter(),
                sample_id=conversation_id,
                config=RollingEpisodeConfig(
                    batch_size=config.episode_batch_size,
                    tail_keep_episode_count=config.episode_tail_keep,
                ),
            )
            builder_name = "locomo"
    else:
        payloads = build_rolling_episodes(
            ordered,
            HeuristicEpisodeSegmenter(),
            sample_id=conversation_id,
            config=RollingEpisodeConfig(
                batch_size=config.episode_batch_size,
                tail_keep_episode_count=config.episode_tail_keep,
            ),
        )
    episodes: list[EpisodeMemory] = []
    for payload in payloads:
        episodes.append(
            EpisodeMemory(
                episode_id=str(payload["episode_id"]),
                conversation_id=conversation_id,
                memory_ids=[str(item) for item in payload.get("memory_ids") or []],
                title=str(payload.get("title") or "")[:80] or None,
                summary=str(payload.get("summary") or ""),
                start_time=_optional_string(payload.get("start_time")),
                end_time=_optional_string(payload.get("end_time")),
                metadata={
                    "builder": builder_name,
                    "topic": payload.get("topic"),
                    "reason": payload.get("reason"),
                    "confidence": payload.get("confidence"),
                    "episode_index": payload.get("episode_index"),
                    "memory_count": payload.get("memory_count"),
                },
            )
        )
    return episodes


def build_episode_segmentation_prompt(memories: list[dict[str, Any]]) -> str:
    lines = []
    for index, memory in enumerate(memories):
        speaker = memory_speaker(memory)
        lines.append(f"mem{index} | speaker={speaker}: {memory_text(memory)}")
    dialogue = "\n".join(lines)
    return f"""Split the following dialogue memories into coherent episodes.

An episode is a locally coherent group of memories around the same event, topic, goal, or temporal situation.

Segmentation rules:
- Prefer topic/event/goal/time-shift boundaries over superficial conversational fluency.
- Keep episode ranges contiguous and non-overlapping.
- Cover every memory exactly once.
- Use the memory ids exactly as shown, such as mem0 and mem12.
- Avoid single-memory episodes unless there is a clear isolated transition.

Dialogue memories:
{dialogue}

Return only plain text lines. Do not return JSON, markdown, bullets, numbering, or extra explanation.
Each line must use exactly this format:
memSTART-memEND | short episode title | stable core discussion topic | brief episode summary | why these memories form one episode | confidence
"""


def parse_segments_from_text(text: str) -> list[EpisodeSegment]:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = extract_json_object(stripped)
        if isinstance(decoded, dict):
            return parse_json_segments(decoded)
    return parse_line_segments(text)


def parse_json_segments(payload: dict[str, Any]) -> list[EpisodeSegment]:
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list):
        return []
    segments: list[EpisodeSegment] = []
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            continue
        start = parse_mem_index(raw.get("start"))
        end = parse_mem_index(raw.get("end"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        segments.append(
            EpisodeSegment(
                start,
                end,
                str(raw.get("title") or "episode").strip(),
                str(raw.get("reason") or "").strip(),
                _clamp_float(raw.get("confidence"), default=0.7),
                str(raw.get("topic") or "").strip(),
                str(raw.get("summary") or "").strip(),
            )
        )
    return segments


def parse_line_segments(text: str) -> list[EpisodeSegment]:
    segments: list[EpisodeSegment] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        match = re.match(r"^mem(\d+)\s*-\s*mem(\d+)\s*\|\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        parts = [part.strip() for part in match.group(3).split("|")]
        if len(parts) < 3:
            continue
        title = parts[0] or "episode"
        if len(parts) >= 5:
            topic, summary, reason = parts[1], parts[2], parts[3]
        elif len(parts) >= 4:
            topic, summary, reason = parts[1], "", parts[2]
        else:
            topic, summary, reason = "", "", parts[1]
        confidence = _parse_confidence(parts[-1])
        if end < start:
            start, end = end, start
        segments.append(EpisodeSegment(start, end, title, reason, confidence, topic, summary))
    return segments


def normalize_segments(segments: list[EpisodeSegment], memory_count: int) -> list[EpisodeSegment]:
    if memory_count <= 0:
        return []
    if not segments:
        return [EpisodeSegment(0, memory_count - 1, "single episode", "fallback full range", 0.0)]
    normalized: list[EpisodeSegment] = []
    cursor = 0
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = max(cursor, min(segment.start, memory_count - 1))
        end = max(start, min(segment.end, memory_count - 1))
        if start > cursor:
            normalized.append(
                EpisodeSegment(cursor, start - 1, "unassigned bridge", "filled uncovered gap", 0.0)
            )
        normalized.append(
            EpisodeSegment(start, end, segment.title, segment.reason, segment.confidence, segment.topic, segment.summary)
        )
        cursor = end + 1
        if cursor >= memory_count:
            break
    if cursor < memory_count:
        normalized.append(EpisodeSegment(cursor, memory_count - 1, "unassigned tail", "filled uncovered tail", 0.0))
    return normalized


def valid_segments(segments: list[EpisodeSegment], memory_count: int) -> bool:
    if memory_count <= 0:
        return not segments
    if not segments:
        return False
    return segments[0].start == 0 and segments[-1].end == memory_count - 1


def parse_mem_index(value: object) -> int | None:
    match = re.fullmatch(r"mem(\d+)", str(value or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _title_from_memories(memories: list[dict[str, Any]]) -> str:
    for item in memories:
        text = memory_text(item)
        if text:
            return text[:80]
    return "episode"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _clamp_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _parse_confidence(text: str) -> float:
    match = re.search(r"([01](?:\.\d+)?)", text)
    if not match:
        return 0.7
    return _clamp_float(match.group(1), default=0.7)
