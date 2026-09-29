"""LoCoMo-style hybrid recall: top-down + bottom-up.

Top-down:  Query -> Trajectory -> Episode -> Memory
Bottom-up: Query -> Memory -> Episode -> Trajectory
Hybrid:    merge both paths by id, keep the higher score.

This ports the research `recall_evidence_pack` algorithm onto cogmem-core
storage (episodes hold `memory_ids`, memories live in a separate collection).
Ranking uses the same BM25 index as trajectory construction. No LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..construction.bm25 import BM25Index
from ..construction.text import compact_text
from ..contracts.recall import ContextMemoryItem
from ..retrieval import query_content_tokens
from ..status import include_superseded_for_query, item_status, query_intent


@dataclass(slots=True)
class SearchHit:
    item_id: str
    score: float
    payload: dict[str, Any]


@dataclass(slots=True)
class MemorySnippet:
    snippet_id: str
    episode_id: str
    trajectory_ids: list[str]
    text: str
    speaker: str | None = None
    time: str | None = None
    score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HybridRecall:
    trajectories: list[SearchHit]
    episodes: list[SearchHit]
    memories: list[MemorySnippet]
    relation_notes: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def hybrid_recall(
    query: str,
    *,
    memories: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    top_trajectories: int = 3,
    top_episodes: int = 5,
    max_snippets: int = 12,
    snippets_per_episode: int = 3,
    global_memory_candidates: int = 8,
    trajectory_strong_threshold: float = 1.0,
    memory_fallback_threshold: float = 1.0,
) -> HybridRecall:
    if not episodes and not trajectories:
        return HybridRecall([], [], [], [], {"retrieval_mode": "hybrid", "reason": "no_structure"})

    episode_by_id = _enrich_episodes(episodes, memories)
    trajectory_by_id = _enrich_trajectories(trajectories)
    episode_to_trajectories = _episode_to_trajectories(list(trajectory_by_id.values()))
    snippets = _build_snippets(episode_by_id, episode_to_trajectories)

    top_down = _recall_top_down(
        query,
        episode_by_id=episode_by_id,
        trajectory_by_id=trajectory_by_id,
        episode_to_trajectories=episode_to_trajectories,
        snippets=snippets,
        top_trajectories=top_trajectories,
        top_episodes=top_episodes,
        max_snippets=max_snippets,
        snippets_per_episode=snippets_per_episode,
        global_memory_candidates=global_memory_candidates,
        trajectory_strong_threshold=trajectory_strong_threshold,
        memory_fallback_threshold=memory_fallback_threshold,
    )
    bottom_up = _recall_bottom_up(
        query,
        episode_by_id=episode_by_id,
        trajectory_by_id=trajectory_by_id,
        episode_to_trajectories=episode_to_trajectories,
        snippets=snippets,
        top_trajectories=top_trajectories,
        top_episodes=top_episodes,
        max_snippets=max_snippets,
        snippets_per_episode=snippets_per_episode,
        global_memory_candidates=global_memory_candidates,
    )
    merged_trajectories = _merge_hits(
        top_down.trajectories,
        bottom_up.trajectories,
        limit=top_trajectories * 2,
    )
    merged_episodes = _merge_hits(
        top_down.episodes,
        bottom_up.episodes,
        limit=top_episodes * 2,
    )
    merged_memories = _merge_snippets(
        top_down.memories,
        bottom_up.memories,
        max_snippets=max_snippets,
        snippets_per_episode=snippets_per_episode,
    )
    notes = _merge_notes(top_down.relation_notes, bottom_up.relation_notes)
    return HybridRecall(
        trajectories=merged_trajectories,
        episodes=merged_episodes,
        memories=merged_memories,
        relation_notes=notes,
        diagnostics={
            "retrieval_mode": "hybrid",
            "top_down": top_down.diagnostics,
            "bottom_up": bottom_up.diagnostics,
            "merged_trajectory_count": len(merged_trajectories),
            "merged_episode_count": len(merged_episodes),
            "merged_memory_count": len(merged_memories),
        },
    )


def assemble_context_items(
    pack: HybridRecall,
    *,
    query: str = "",
    include_superseded: bool = False,
    current_limit: int = 2,
    related_limit: int = 6,
    evidence_limit: int = 8,
) -> tuple[list[ContextMemoryItem], list[ContextMemoryItem], list[ContextMemoryItem], list[ContextMemoryItem]]:
    keep_superseded = include_superseded_for_query(query, include_superseded=include_superseded)
    background = [_trajectory_item(hit) for hit in pack.trajectories if hit.score > 0]
    primary: list[ContextMemoryItem] = []
    related: list[ContextMemoryItem] = []
    for hit in pack.episodes:
        if hit.score <= 0:
            continue
        item = _episode_item(hit)
        role = str(hit.payload.get("_retrieval_role") or "primary")
        status = item_status(hit.payload)
        outdated = status == "superseded" or role == "updated_background"
        if outdated and not keep_superseded:
            if len(related) < related_limit:
                item.source = "related_experience"
                related.append(item)
            continue
        if role in {"primary", "latest_update", "forced_detail"} and len(primary) < current_limit:
            primary.append(item)
        elif len(related) < related_limit:
            item.source = "related_experience"
            related.append(item)
    evidence = [
        _snippet_item(snippet)
        for snippet in pack.memories
        if snippet.score > 0
        and (keep_superseded or item_status(snippet.payload) != "superseded")
    ][:evidence_limit]
    return background, primary, related, evidence


def _recall_top_down(
    query: str,
    *,
    episode_by_id: dict[str, dict[str, Any]],
    trajectory_by_id: dict[str, dict[str, Any]],
    episode_to_trajectories: dict[str, list[str]],
    snippets: list[MemorySnippet],
    top_trajectories: int,
    top_episodes: int,
    max_snippets: int,
    snippets_per_episode: int,
    global_memory_candidates: int,
    trajectory_strong_threshold: float,
    memory_fallback_threshold: float,
) -> HybridRecall:
    fuzzy_records = [
        (trajectory_id, _trajectory_index_text(payload, episode_by_id), payload)
        for trajectory_id, payload in trajectory_by_id.items()
    ]
    trajectory_hits = _rank_records(query, fuzzy_records, top_k=top_trajectories)
    fallback_hit: SearchHit | None = None
    fallback_added = False
    fallback_candidates: list[SearchHit] = []
    fallback_should_run = _should_use_memory_fallback(
        trajectory_hits, strong_threshold=trajectory_strong_threshold
    )
    if fallback_should_run:
        fallback_candidates = _rank_snippet_hits(query, snippets, top_k=global_memory_candidates)
        for hit in fallback_candidates:
            if hit.score >= memory_fallback_threshold:
                fallback_hit = hit
                before = {item.item_id for item in trajectory_hits}
                trajectory_hits = _add_forced_trajectory(
                    trajectory_hits,
                    fallback_hit=hit,
                    episode_by_id=episode_by_id,
                    trajectory_by_id=trajectory_by_id,
                    episode_to_trajectories=episode_to_trajectories,
                    max_trajectories=top_trajectories + 1,
                )
                fallback_added = {item.item_id for item in trajectory_hits} != before
                break

    candidate_episode_ids = {
        str(episode_id)
        for hit in trajectory_hits
        for episode_id in (hit.payload.get("episode_ids") or [])
    }
    if not candidate_episode_ids:
        candidate_episode_ids = set(episode_by_id)
    episode_records = [
        (episode_id, _episode_index_text(payload), payload)
        for episode_id, payload in episode_by_id.items()
        if episode_id in candidate_episode_ids
    ]
    episode_hits = _rank_records(query, episode_records, top_k=top_episodes)
    selected_ids, notes, roles = _arbitrate_relations(
        query, trajectory_hits, episode_hits, episode_by_id
    )
    if fallback_hit is not None:
        snippet: MemorySnippet = fallback_hit.payload["snippet"]
        if snippet.episode_id in episode_by_id:
            selected_ids.add(snippet.episode_id)
            roles.setdefault(snippet.episode_id, "forced_detail")
    episode_hits = _enrich_episode_hits(
        query, episode_hits, selected_ids, roles, episode_by_id, top_k=top_episodes
    )
    evidence = _rank_snippets(
        query,
        snippets,
        selected_episode_ids={hit.item_id for hit in episode_hits},
        max_snippets=max_snippets,
        snippets_per_episode=snippets_per_episode,
    )
    return HybridRecall(
        trajectories=trajectory_hits,
        episodes=episode_hits,
        memories=evidence,
        relation_notes=notes,
        diagnostics={
            "retrieval_mode": "top_down",
            "raw_memory_fallback_considered": fallback_should_run,
            "raw_memory_fallback_added": fallback_added,
            "selected_trajectory_count": len(trajectory_hits),
            "selected_episode_count": len(episode_hits),
        },
    )


def _recall_bottom_up(
    query: str,
    *,
    episode_by_id: dict[str, dict[str, Any]],
    trajectory_by_id: dict[str, dict[str, Any]],
    episode_to_trajectories: dict[str, list[str]],
    snippets: list[MemorySnippet],
    top_trajectories: int,
    top_episodes: int,
    max_snippets: int,
    snippets_per_episode: int,
    global_memory_candidates: int,
) -> HybridRecall:
    memory_hits = _rank_snippet_hits(
        query, snippets, top_k=max(global_memory_candidates, max_snippets, top_episodes)
    )
    selected_episode_ids: list[str] = []
    for hit in memory_hits:
        snippet: MemorySnippet = hit.payload["snippet"]
        if snippet.episode_id not in selected_episode_ids:
            selected_episode_ids.append(snippet.episode_id)
        if len(selected_episode_ids) >= top_episodes:
            break
    selected_set = set(selected_episode_ids)
    episode_scores: dict[str, float] = {}
    for hit in memory_hits:
        snippet = hit.payload["snippet"]
        if snippet.episode_id not in selected_set:
            continue
        episode_scores[snippet.episode_id] = max(
            episode_scores.get(snippet.episode_id, 0.0), hit.score
        )
    episode_hits = sorted(
        [
            SearchHit(
                item_id=episode_id,
                score=score,
                payload={**episode_by_id[episode_id], "_retrieval_role": "primary"},
            )
            for episode_id, score in episode_scores.items()
            if episode_id in episode_by_id
        ],
        key=lambda hit: (-hit.score, hit.item_id),
    )
    trajectory_hits = _trajectories_from_memory_hits(
        memory_hits,
        selected_episode_ids=selected_set,
        episode_by_id=episode_by_id,
        trajectory_by_id=trajectory_by_id,
        episode_to_trajectories=episode_to_trajectories,
        top_k=top_trajectories,
    )
    selected_ids, notes, roles = _arbitrate_relations(
        query, trajectory_hits, episode_hits, episode_by_id
    )
    episode_hits = _enrich_episode_hits(
        query, episode_hits, selected_ids, roles, episode_by_id, top_k=top_episodes
    )
    evidence = _rank_snippets(
        query,
        snippets,
        selected_episode_ids={hit.item_id for hit in episode_hits},
        max_snippets=max_snippets,
        snippets_per_episode=snippets_per_episode,
    )
    return HybridRecall(
        trajectories=trajectory_hits,
        episodes=episode_hits,
        memories=evidence,
        relation_notes=notes,
        diagnostics={
            "retrieval_mode": "bottom_up",
            "raw_memory_candidate_count": len(memory_hits),
            "selected_episode_count": len(episode_hits),
            "selected_trajectory_count": len(trajectory_hits),
        },
    )


def _enrich_episodes(
    episodes: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    memory_by_id = {
        str(item.get("memory_id") or ""): item
        for item in memories
        if item.get("memory_id")
    }
    enriched: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            continue
        metadata = dict(episode.get("metadata") or {})
        attached = [
            memory_by_id[memory_id]
            for memory_id in (episode.get("memory_ids") or [])
            if memory_id in memory_by_id
        ]
        payload = dict(episode)
        payload["memories"] = attached
        payload["topic"] = payload.get("topic") or metadata.get("topic") or payload.get("title") or ""
        payload["reason"] = payload.get("reason") or metadata.get("reason") or ""
        payload["episode_index"] = metadata.get("episode_index")
        enriched[episode_id] = payload
    return enriched


def _enrich_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}
    for trajectory in trajectories:
        trajectory_id = str(trajectory.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        metadata = dict(trajectory.get("metadata") or {})
        payload = dict(trajectory)
        payload["topic"] = payload.get("topic") or metadata.get("topic") or payload.get("title") or ""
        payload["reason"] = payload.get("reason") or metadata.get("reason") or ""
        enriched[trajectory_id] = payload
    return enriched


def _episode_to_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for trajectory in trajectories:
        trajectory_id = str(trajectory.get("trajectory_id") or "")
        for episode_id in trajectory.get("episode_ids") or []:
            mapping.setdefault(str(episode_id), []).append(trajectory_id)
    return mapping


def _build_snippets(
    episode_by_id: dict[str, dict[str, Any]],
    episode_to_trajectories: dict[str, list[str]],
) -> list[MemorySnippet]:
    snippets: list[MemorySnippet] = []
    for episode_id, episode in episode_by_id.items():
        for index, memory in enumerate(episode.get("memories") or []):
            text = compact_text(str(memory.get("text") or memory.get("content") or ""))
            if not text:
                continue
            memory_id = str(memory.get("memory_id") or f"mem{index}")
            metadata = memory.get("metadata") or {}
            snippets.append(
                MemorySnippet(
                    snippet_id=memory_id,
                    episode_id=episode_id,
                    trajectory_ids=list(episode_to_trajectories.get(episode_id, [])),
                    text=text,
                    speaker=_optional_string(metadata.get("speaker")),
                    time=_optional_string(memory.get("timestamp")),
                    payload=dict(memory),
                )
            )
    return snippets


def _trajectory_index_text(
    trajectory: dict[str, Any],
    episode_by_id: dict[str, dict[str, Any]],
    *,
    max_member_episodes: int = 5,
) -> str:
    member_texts = []
    for episode_id in (trajectory.get("episode_ids") or [])[:max_member_episodes]:
        episode = episode_by_id.get(str(episode_id))
        if episode:
            member_texts.append(_episode_index_text(episode, max_memory_chars=240))
    parts = _unique_labeled_parts(
        [
            ("title", trajectory.get("title")),
            ("topic", trajectory.get("topic")),
            ("summary", trajectory.get("summary")),
            ("reason", trajectory.get("reason")),
        ]
    )
    if member_texts:
        parts.append("\n".join(member_texts))
    return "\n".join(parts)


def _episode_index_text(episode: dict[str, Any], *, max_memory_chars: int = 800) -> str:
    memories = [
        compact_text(str(memory.get("text") or ""))
        for memory in episode.get("memories") or []
        if isinstance(memory, dict) and memory.get("text")
    ]
    memory_text = compact_text(" ".join(memories))
    if len(memory_text) > max_memory_chars:
        memory_text = memory_text[:max_memory_chars].rsplit(" ", 1)[0]
    parts = _unique_labeled_parts(
        [
            ("title", episode.get("title")),
            ("topic", episode.get("topic")),
            ("summary", episode.get("summary")),
            ("reason", episode.get("reason")),
        ]
    )
    if memory_text:
        parts.append(f"memories: {memory_text}")
    return "\n".join(parts)


def _rank_records(
    query: str,
    records: list[tuple[str, str, dict[str, Any]]],
    *,
    top_k: int,
) -> list[SearchHit]:
    if not records or top_k <= 0:
        return []
    bm25_query = _content_query(query)
    index = BM25Index([text for _, text, _ in records])
    hits = [
        SearchHit(
            item_id=item_id,
            score=index.score(bm25_query, record_index),
            payload=payload,
        )
        for record_index, (item_id, _, payload) in enumerate(records)
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.item_id))[:top_k]


def _content_query(query: str) -> str:
    tokens = query_content_tokens(query)
    return " ".join(sorted(tokens, key=lambda token: (-len(token), token))) or query


def _unique_labeled_parts(fields: list[tuple[str, Any]]) -> list[str]:
    seen: set[str] = set()
    parts: list[str] = []
    for label, value in fields:
        text = compact_text(str(value or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(f"{label}: {text}")
    return parts


def _rank_snippet_hits(query: str, snippets: list[MemorySnippet], *, top_k: int) -> list[SearchHit]:
    records = [
        (snippet.snippet_id, _snippet_index_text(snippet), {"snippet": snippet})
        for snippet in snippets
    ]
    return _rank_records(query, records, top_k=top_k)


def _rank_snippets(
    query: str,
    snippets: list[MemorySnippet],
    *,
    selected_episode_ids: set[str],
    max_snippets: int,
    snippets_per_episode: int,
) -> list[MemorySnippet]:
    candidates = [snippet for snippet in snippets if snippet.episode_id in selected_episode_ids]
    if not include_superseded_for_query(query):
        current = [snippet for snippet in candidates if item_status(snippet.payload) != "superseded"]
        if current:
            candidates = current
    hits = _rank_snippet_hits(query, candidates, top_k=len(candidates))
    selected: list[MemorySnippet] = []
    counts: dict[str, int] = {}
    for hit in hits:
        snippet: MemorySnippet = hit.payload["snippet"]
        if hit.score <= 0:
            continue
        if counts.get(snippet.episode_id, 0) >= snippets_per_episode:
            continue
        selected.append(
            MemorySnippet(
                snippet_id=snippet.snippet_id,
                episode_id=snippet.episode_id,
                trajectory_ids=snippet.trajectory_ids,
                text=snippet.text,
                speaker=snippet.speaker,
                time=snippet.time,
                score=hit.score,
                payload=snippet.payload,
            )
        )
        counts[snippet.episode_id] = counts.get(snippet.episode_id, 0) + 1
        if len(selected) >= max_snippets:
            break
    return selected


def _snippet_index_text(snippet: MemorySnippet) -> str:
    parts = [
        f"speaker: {snippet.speaker or ''}",
        f"time: {snippet.time or ''}",
        f"text: {snippet.text}",
    ]
    return "\n".join(part for part in parts if compact_text(part.split(":", 1)[-1]))


def _should_use_memory_fallback(trajectory_hits: list[SearchHit], *, strong_threshold: float) -> bool:
    if not trajectory_hits:
        return True
    return trajectory_hits[0].score < strong_threshold


def _add_forced_trajectory(
    trajectory_hits: list[SearchHit],
    *,
    fallback_hit: SearchHit,
    episode_by_id: dict[str, dict[str, Any]],
    trajectory_by_id: dict[str, dict[str, Any]],
    episode_to_trajectories: dict[str, list[str]],
    max_trajectories: int,
) -> list[SearchHit]:
    snippet: MemorySnippet = fallback_hit.payload["snippet"]
    trajectory_ids = episode_to_trajectories.get(snippet.episode_id) or []
    if trajectory_ids:
        trajectory_id = trajectory_ids[0]
        trajectory = trajectory_by_id.get(trajectory_id)
        if trajectory is None:
            return trajectory_hits
        payload = {**trajectory, "_recall_mode": "forced_detail"}
    else:
        episode = episode_by_id.get(snippet.episode_id)
        if episode is None:
            return trajectory_hits
        trajectory_id = f"runtime_detail_{snippet.episode_id}"
        payload = {
            "trajectory_id": trajectory_id,
            "title": episode.get("title") or "Runtime detail trajectory",
            "topic": episode.get("topic") or episode.get("title") or "",
            "summary": episode.get("summary") or "",
            "episode_ids": [snippet.episode_id],
            "relations": [],
            "_recall_mode": "forced_detail",
            "_runtime_pseudo": True,
        }
    updated: list[SearchHit] = []
    replaced = False
    for hit in trajectory_hits:
        if hit.item_id == trajectory_id:
            updated.append(SearchHit(hit.item_id, max(hit.score, fallback_hit.score), payload))
            replaced = True
        else:
            updated.append(hit)
    if not replaced:
        updated.append(SearchHit(trajectory_id, fallback_hit.score, payload))
    return sorted(updated, key=lambda hit: (-hit.score, hit.item_id))[:max_trajectories]


def _trajectories_from_memory_hits(
    memory_hits: list[SearchHit],
    *,
    selected_episode_ids: set[str],
    episode_by_id: dict[str, dict[str, Any]],
    trajectory_by_id: dict[str, dict[str, Any]],
    episode_to_trajectories: dict[str, list[str]],
    top_k: int,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for hit in memory_hits:
        snippet: MemorySnippet = hit.payload["snippet"]
        if snippet.episode_id not in selected_episode_ids:
            continue
        trajectory_ids = episode_to_trajectories.get(snippet.episode_id) or []
        if trajectory_ids:
            for trajectory_id in trajectory_ids:
                trajectory = trajectory_by_id.get(trajectory_id)
                if trajectory is None:
                    continue
                scores[trajectory_id] = max(scores.get(trajectory_id, 0.0), hit.score)
                payloads[trajectory_id] = {**trajectory, "_recall_mode": "bottom_up"}
        else:
            episode = episode_by_id.get(snippet.episode_id)
            if episode is None:
                continue
            trajectory_id = f"runtime_detail_{snippet.episode_id}"
            scores[trajectory_id] = max(scores.get(trajectory_id, 0.0), hit.score)
            payloads[trajectory_id] = {
                "trajectory_id": trajectory_id,
                "title": episode.get("title") or "Runtime detail trajectory",
                "summary": episode.get("summary") or "",
                "episode_ids": [snippet.episode_id],
                "relations": [],
                "_recall_mode": "bottom_up",
                "_runtime_pseudo": True,
            }
    hits = [
        SearchHit(item_id=trajectory_id, score=score, payload=payloads[trajectory_id])
        for trajectory_id, score in scores.items()
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.item_id))[:top_k]


def _arbitrate_relations(
    query: str,
    trajectory_hits: list[SearchHit],
    episode_hits: list[SearchHit],
    episode_by_id: dict[str, dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]], dict[str, str]]:
    intent = _query_intent(query)
    selected = {hit.item_id for hit in episode_hits}
    roles = {hit.item_id: "primary" for hit in episode_hits}
    notes: list[dict[str, Any]] = []
    for trajectory_hit in trajectory_hits:
        trajectory = trajectory_hit.payload
        member_ids = {str(value) for value in trajectory.get("episode_ids") or []}
        selected_in_trajectory = selected & member_ids
        if not selected_in_trajectory:
            continue
        for relation in trajectory.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            source_id = str(relation.get("source_episode_id") or "")
            target_id = str(relation.get("target_episode_id") or "")
            if source_id not in member_ids or target_id not in member_ids:
                continue
            if source_id not in selected_in_trajectory and target_id not in selected_in_trajectory:
                continue
            relation_type = str(relation.get("relation") or "continuation").lower()
            added = _relation_expansion_ids(
                relation_type, source_id, target_id, selected_in_trajectory, intent
            )
            for episode_id in added:
                if episode_id in episode_by_id:
                    selected.add(episode_id)
                    roles[episode_id] = _relation_role(
                        relation_type, episode_id, source_id, target_id, intent
                    )
            notes.append(
                {
                    "trajectory_id": trajectory_hit.item_id,
                    "source_episode_id": source_id,
                    "target_episode_id": target_id,
                    "relation": relation_type,
                    "query_intent": intent,
                    "expanded_episode_ids": sorted(added),
                }
            )
    return selected, notes, roles


def _enrich_episode_hits(
    query: str,
    episode_hits: list[SearchHit],
    selected_episode_ids: set[str],
    role_by_episode: dict[str, str],
    episode_by_id: dict[str, dict[str, Any]],
    *,
    top_k: int,
) -> list[SearchHit]:
    intent = _query_intent(query)
    keep_superseded = include_superseded_for_query(query)
    enriched = [
        SearchHit(
            hit.item_id,
            _status_adjusted_score(
                hit.score,
                role_by_episode.get(hit.item_id, "primary"),
                item_status(hit.payload),
                intent,
            ),
            {**hit.payload, "_retrieval_role": role_by_episode.get(hit.item_id, "primary")},
        )
        for hit in episode_hits
    ]
    existing = {hit.item_id for hit in enriched}
    missing = [
        (episode_id, _episode_index_text(episode_by_id[episode_id]), episode_by_id[episode_id])
        for episode_id in selected_episode_ids - existing
        if episode_id in episode_by_id
    ]
    for hit in _rank_records(query, missing, top_k=len(missing)):
        role = role_by_episode.get(hit.item_id, "supporting")
        enriched.append(
            SearchHit(
                hit.item_id,
                _status_adjusted_score(hit.score, role, item_status(hit.payload), intent),
                {**hit.payload, "_retrieval_role": role},
            )
        )
    if not keep_superseded:
        current = [
            hit
            for hit in enriched
            if item_status(hit.payload) != "superseded"
            or role_by_episode.get(hit.item_id) == "latest_update"
        ]
        if current:
            enriched = current
    limit = max(1, top_k, len(selected_episode_ids))
    return sorted(enriched, key=lambda hit: (-hit.score, hit.item_id))[:limit]


def _merge_hits(first: list[SearchHit], second: list[SearchHit], *, limit: int) -> list[SearchHit]:
    merged: dict[str, SearchHit] = {}
    sources: dict[str, set[str]] = {}
    for label, hits in (("top_down", first), ("bottom_up", second)):
        for hit in hits:
            existing = merged.get(hit.item_id)
            if existing is None or hit.score > existing.score:
                merged[hit.item_id] = hit
            sources.setdefault(hit.item_id, set()).add(label)
    enriched = [
        SearchHit(
            hit.item_id,
            hit.score,
            {**hit.payload, "_retrieval_sources": sorted(sources.get(hit.item_id, set()))},
        )
        for hit in merged.values()
    ]
    return sorted(enriched, key=lambda hit: (-hit.score, hit.item_id))[:limit]


def _merge_snippets(
    first: list[MemorySnippet],
    second: list[MemorySnippet],
    *,
    max_snippets: int,
    snippets_per_episode: int,
) -> list[MemorySnippet]:
    merged: dict[str, MemorySnippet] = {}
    for snippet in first + second:
        existing = merged.get(snippet.snippet_id)
        if existing is None or snippet.score > existing.score:
            merged[snippet.snippet_id] = snippet
    selected: list[MemorySnippet] = []
    counts: dict[str, int] = {}
    for snippet in sorted(merged.values(), key=lambda item: (-item.score, item.snippet_id)):
        if snippet.score <= 0:
            continue
        if counts.get(snippet.episode_id, 0) >= snippets_per_episode:
            continue
        selected.append(snippet)
        counts[snippet.episode_id] = counts.get(snippet.episode_id, 0) + 1
        if len(selected) >= max_snippets:
            break
    return selected


def _merge_notes(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for note in first + second:
        key = (
            str(note.get("trajectory_id") or ""),
            str(note.get("source_episode_id") or ""),
            str(note.get("target_episode_id") or ""),
            str(note.get("relation") or ""),
        )
        merged.setdefault(key, note)
    return list(merged.values())


def _query_intent(query: str) -> str:
    return query_intent(query)


def _status_adjusted_score(score: float, role: str, status: str, intent: str) -> float:
    adjusted = float(score)
    if role == "latest_update" or (status == "active" and role in {"primary", "forced_detail"}):
        if intent == "current_state":
            adjusted *= 1.35
    if role == "updated_background" or status == "superseded":
        if intent == "current_state":
            adjusted *= 0.25
        elif intent != "history_process":
            adjusted *= 0.45
    return adjusted


def _relation_expansion_ids(
    relation_type: str,
    source_id: str,
    target_id: str,
    selected_ids: set[str],
    query_intent: str,
) -> set[str]:
    if relation_type in {"update", "correction"}:
        if query_intent == "current_state":
            return {target_id}
        return {source_id, target_id}
    if relation_type == "conflict":
        return {source_id, target_id}
    if relation_type == "progression":
        if source_id in selected_ids:
            return {target_id}
        if target_id in selected_ids:
            return {source_id}
    return set()


def _relation_role(
    relation_type: str,
    episode_id: str,
    source_id: str,
    target_id: str,
    query_intent: str,
) -> str:
    if relation_type in {"update", "correction"}:
        return "latest_update" if episode_id == target_id else "updated_background"
    if relation_type == "conflict":
        return "conflict"
    if relation_type == "progression":
        return "supporting" if query_intent != "history_process" else "primary"
    return "supporting"


def _prompt_span(*parts: Any) -> str:
    unique: list[str] = []
    for part in parts:
        text = compact_text(str(part or ""))
        text = re.sub(r"\b(?:tool_call|tool_result)\b[^\n。]*|path=/[^\s]+", " ", text, flags=re.I)
        text = compact_text(text)
        if not text:
            continue
        if unique and (text == unique[-1] or text.startswith(unique[-1]) or unique[-1] in text):
            unique[-1] = text if len(text) >= len(unique[-1]) else unique[-1]
            continue
        unique.append(text)
    return compact_text(" ".join(unique))


def _trajectory_item(hit: SearchHit) -> ContextMemoryItem:
    text = _prompt_span(
        hit.payload.get("title"),
        hit.payload.get("topic"),
        hit.payload.get("summary"),
    ) or str(hit.payload.get("reason") or hit.item_id)
    return ContextMemoryItem(
        item_id=hit.item_id,
        text=text,
        source="long_term_background",
        score=hit.score,
        metadata={
            "episode_ids": list(hit.payload.get("episode_ids") or []),
            "recall_mode": hit.payload.get("_recall_mode"),
            "retrieval_sources": hit.payload.get("_retrieval_sources") or [],
        },
    )


def _episode_item(hit: SearchHit, *, source: str = "current_situation") -> ContextMemoryItem:
    text = _prompt_span(hit.payload.get("title"), hit.payload.get("summary")) or hit.item_id
    return ContextMemoryItem(
        item_id=hit.item_id,
        text=text,
        source=source,
        score=hit.score,
        timestamp=_optional_string(hit.payload.get("start_time") or hit.payload.get("end_time")),
        metadata={
            "memory_ids": list(hit.payload.get("memory_ids") or []),
            "retrieval_role": hit.payload.get("_retrieval_role"),
            "retrieval_sources": hit.payload.get("_retrieval_sources") or [],
            "status": item_status(hit.payload),
            "cognitive_status": item_status(hit.payload),
            "superseded_by": (hit.payload.get("metadata") or {}).get("superseded_by"),
        },
    )


def _snippet_item(snippet: MemorySnippet) -> ContextMemoryItem:
    return ContextMemoryItem(
        item_id=snippet.snippet_id,
        text=snippet.text,
        source="supporting_evidence",
        score=snippet.score,
        timestamp=snippet.time,
        metadata={
            "episode_id": snippet.episode_id,
            "trajectory_ids": snippet.trajectory_ids,
            "speaker": snippet.speaker,
            "status": item_status(snippet.payload),
            "cognitive_status": item_status(snippet.payload),
        },
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = compact_text(str(value))
    return text or None
