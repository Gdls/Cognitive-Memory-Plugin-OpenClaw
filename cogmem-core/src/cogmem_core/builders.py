from __future__ import annotations

from hashlib import sha1
from typing import Any

from .construction.config import ConstructionConfig
from .events import CogMemEvent
from .schemas import EpisodeMemory, MemoryObject, TrajectoryMemory


def build_memory_objects_from_events(
    conversation_id: str,
    events: list[CogMemEvent],
) -> list[MemoryObject]:
    memories: list[MemoryObject] = []
    for event in events:
        text = event.summary or event.text or summarize_event(event)
        if not text:
            continue
        memories.append(
            MemoryObject(
                memory_id=stable_memory_id(conversation_id, event.event_id, text),
                conversation_id=conversation_id,
                source_event_id=event.event_id,
                text=text,
                memory_type=guess_memory_type(event),
                timestamp=event.timestamp,
                metadata={
                    "event_type": event.event_type,
                    "speaker": event.speaker,
                    "tool_name": event.tool_name,
                    "status": event.status,
                    "store_policy": event.store_policy,
                },
            )
        )
    return memories


def summarize_event(event: CogMemEvent) -> str:
    if event.event_type in {"tool_call", "tool_observation"}:
        parts = [event.event_type]
        if event.tool_name:
            parts.append(event.tool_name)
        if event.status:
            parts.append(event.status)
        return " ".join(parts)
    return event.text


def guess_memory_type(event: CogMemEvent) -> str:
    if event.event_type in {"tool_call", "tool_observation", "execution_summary"}:
        return "event"
    if event.event_type == "system_event":
        return "rule"
    return "event"


def stable_memory_id(conversation_id: str, event_id: str, text: str) -> str:
    digest = sha1(f"{conversation_id}:{event_id}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"mem_{digest}"


def merge_unique_memories(existing: list[dict[str, Any]], new_items: list[MemoryObject]) -> list[dict[str, Any]]:
    by_id = {str(item.get("memory_id")): dict(item) for item in existing if item.get("memory_id")}
    for item in new_items:
        by_id[item.memory_id] = item.to_dict()
    return list(by_id.values())


def build_episode_memories_from_objects(
    conversation_id: str,
    memories: list[dict[str, Any]],
    *,
    max_memories_per_episode: int = 8,
    config: ConstructionConfig | None = None,
) -> list[EpisodeMemory]:
    resolved = config or ConstructionConfig()
    if resolved.resolved_episode_builder() != "placeholder":
        from .construction.episodes import locomo_episodes_from_objects

        return locomo_episodes_from_objects(conversation_id, memories, resolved)
    episodes: list[EpisodeMemory] = []
    chunk_size = max(1, max_memories_per_episode)
    ordered_memories = [memory for memory in memories if memory.get("memory_id")]
    for index in range(0, len(ordered_memories), chunk_size):
        chunk = ordered_memories[index : index + chunk_size]
        memory_ids = [str(memory["memory_id"]) for memory in chunk]
        text_parts = [str(memory.get("text") or "") for memory in chunk if memory.get("text")]
        if not text_parts:
            continue
        summary = summarize_texts(text_parts)
        episodes.append(
            EpisodeMemory(
                episode_id=stable_episode_id(conversation_id, memory_ids),
                conversation_id=conversation_id,
                memory_ids=memory_ids,
                title=summary[:80],
                summary=summary,
                start_time=optional_string(chunk[0].get("timestamp")),
                end_time=optional_string(chunk[-1].get("timestamp")),
                metadata={
                    "builder": "chunked_memory_objects",
                    "memory_count": len(memory_ids),
                },
            )
        )
    return episodes


def build_trajectory_memories_from_episodes(
    conversation_id: str,
    episodes: list[dict[str, Any] | EpisodeMemory],
    *,
    config: ConstructionConfig | None = None,
) -> list[TrajectoryMemory]:
    normalized = [
        episode.to_dict() if isinstance(episode, EpisodeMemory) else dict(episode)
        for episode in episodes
        if (episode.episode_id if isinstance(episode, EpisodeMemory) else episode.get("episode_id"))
    ]
    resolved = config or ConstructionConfig()
    if resolved.resolved_trajectory_builder() != "placeholder":
        from .construction.trajectories import locomo_trajectories_from_episodes

        return locomo_trajectories_from_episodes(conversation_id, normalized, resolved)
    if not normalized:
        return []
    episode_ids = [str(episode["episode_id"]) for episode in normalized]
    summaries = [str(episode.get("summary") or "") for episode in normalized if episode.get("summary")]
    summary = summarize_texts(summaries, max_chars=360)
    relations = [
        {
            "source_episode_id": source,
            "target_episode_id": target,
            "relation": "continuation",
            "reason": "Sequential episodes from the same conversation build a default runtime trajectory.",
        }
        for source, target in zip(episode_ids, episode_ids[1:])
    ]
    return [
        TrajectoryMemory(
            trajectory_id=stable_trajectory_id(conversation_id, episode_ids),
            conversation_id=conversation_id,
            episode_ids=episode_ids,
            title=f"{conversation_id} runtime trajectory",
            summary=summary,
            relations=relations,
            metadata={
                "builder": "conversation_episode_sequence",
                "episode_count": len(episode_ids),
            },
        )
    ]


def merge_unique_payloads(
    existing: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    by_id = {str(item.get(key)): dict(item) for item in existing if item.get(key)}
    for item in new_items:
        if item.get(key):
            by_id[str(item[key])] = dict(item)
    return list(by_id.values())


def stable_episode_id(conversation_id: str, memory_ids: list[str]) -> str:
    digest = sha1(f"{conversation_id}:episode:{','.join(memory_ids)}".encode("utf-8")).hexdigest()[:12]
    return f"ep_{digest}"


def stable_trajectory_id(conversation_id: str, episode_ids: list[str]) -> str:
    digest = sha1(f"{conversation_id}:trajectory:{','.join(episode_ids)}".encode("utf-8")).hexdigest()[:12]
    return f"trj_{digest}"


def summarize_texts(texts: list[str], *, max_chars: int = 240) -> str:
    summary = " ".join(text.strip() for text in texts if text.strip())
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rsplit(" ", 1)[0].strip() or summary[:max_chars]


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
