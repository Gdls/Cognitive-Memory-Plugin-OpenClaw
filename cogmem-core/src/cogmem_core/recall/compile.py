"""Compile retrieved structure into a short, question-driven Cognitive Context.

Internal recall can keep many trajectories/episodes. The prompt only gets items
that are relevant to the query (or to active preferences on identity/style
questions), compressed to a token budget. Preference Memory stays guidance,
never factual evidence.

Later update/correction statements in the same trajectory replace earlier ones
in Current Situation; conflicts stay as a pair with a short note.
"""

from __future__ import annotations

import re
from typing import Any

from .bidirectional import HybridRecall, assemble_context_items
from ..contracts.recall import CognitiveContext, ContextMemoryItem
from ..retrieval import query_content_tokens
from ..status import include_superseded_for_query, item_status, query_intent
from .progressive import question_layer

GUIDANCE_MARKERS = (
    "怎么叫",
    "叫你",
    "称呼",
    "人设",
    "偏好",
    "自称",
    "谁啊",
    "who are you",
    "what should i call",
)
CALL_MARKERS = ("怎么叫", "叫你", "称呼", "自称", "what should i call", "call you")
UPDATE_RELATIONS = {"update", "correction"}
REVISION_MARKERS = ("不要", "不用", "别再", "不再", "不碰", "改成", "纠正", "其实是")
TOOL_NOISE = re.compile(
    r"\b(?:tool_call|tool_result)\b[^\n。]*|path=/[^\s]+|success:\s*\S+",
    re.I,
)
ITEM_CHAR_LIMIT = {
    "preference_memory": 160,
    "long_term_background": 140,
    "current_situation": 180,
    "related_experience": 140,
    "supporting_evidence": 160,
    "temporal_memory": 120,
    "execution_memory": 120,
}
SLOT_LIMITS = {
    "preference_memory": 4,
    "long_term_background": 2,
    "current_situation": 2,
    "related_experience": 2,
    "supporting_evidence": 4,
    "temporal_memory": 3,
    "execution_memory": 2,
}


def is_guidance_query(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in GUIDANCE_MARKERS)


def text_overlaps_tokens(text: str, tokens: set[str]) -> bool:
    return _overlaps(text, tokens)


def compile_cognitive_context(
    *,
    query: str,
    conversation_id: str,
    pack: HybridRecall,
    preference_hits: list[ContextMemoryItem],
    temporal_hits: list[ContextMemoryItem],
    execution_hits: list[ContextMemoryItem],
    episodes: list[dict[str, Any]] | None = None,
    trajectories: list[dict[str, Any]] | None = None,
    token_budget: int | None = None,
    include_superseded: bool = False,
) -> CognitiveContext:
    episode_list = episodes or []
    trajectory_list = trajectories or []
    keep_superseded = include_superseded_for_query(query, include_superseded=include_superseded)
    preference_hits, pref_superseded = _arbitrate_preferences(
        preference_hits, episode_list, keep_superseded=keep_superseded
    )

    background, situation, related, evidence = assemble_context_items(
        pack,
        query=query,
        include_superseded=keep_superseded,
        current_limit=2,
        related_limit=4,
        evidence_limit=8,
    )
    dropped = 0
    guidance = is_guidance_query(query)
    if guidance:
        lifted_bg, lifted_sit, lifted_rel = _lift_preference_structure(
            query, preference_hits, episode_list, trajectory_list
        )
        background, dropped = _merge_unique(background, lifted_bg, dropped)
        situation, dropped = _merge_unique(situation, lifted_sit, dropped)
        related, dropped = _merge_unique(related, lifted_rel, dropped)

    query_tokens = query_content_tokens(query)
    keep_tokens = query_tokens
    if guidance:
        keep_tokens = query_tokens | _preference_tokens(preference_hits)
    layer = question_layer(query)
    if layer == "trajectory":
        background, dropped = _merge_unique(
            background, _lift_trajectories(trajectory_list), dropped
        )
        background, dropped = _filter_slot(
            background, keep_tokens, dropped, require_overlap=False
        )
    else:
        background, dropped = _filter_slot(
            background, keep_tokens, dropped, require_overlap=True
        )
    situation, dropped = _filter_slot(situation, keep_tokens, dropped, require_overlap=bool(keep_tokens))
    related, dropped = _filter_slot(related, keep_tokens, dropped, require_overlap=True)
    evidence, dropped = _filter_slot(evidence, keep_tokens, dropped, require_overlap=True)
    temporal_hits, dropped = _filter_slot(temporal_hits, keep_tokens, dropped, require_overlap=True)

    evidence = _drop_if_already_in(evidence, preference_hits)
    if not keep_superseded:
        evidence = [item for item in evidence if item_status(item.metadata) != "superseded"] or evidence
    situation, related, arbitration = _arbitrate_structure(
        situation, related, pack, trajectory_list
    )

    conflict_notes = [
        note
        for note in pack.relation_notes
        if str(note.get("relation") or "").lower() in {"conflict", "correction", "update"}
    ]

    context = CognitiveContext(
        query=query,
        conversation_id=conversation_id,
        preference_memory=_compress_slot(preference_hits, "preference_memory"),
        long_term_background=_compress_slot(background, "long_term_background"),
        current_situation=_compress_slot(situation, "current_situation"),
        related_experience=_compress_slot(related, "related_experience"),
        supporting_evidence=_compress_slot(evidence, "supporting_evidence"),
        temporal_memory=_compress_slot(temporal_hits, "temporal_memory"),
        execution_memory=_compress_slot(execution_hits, "execution_memory"),
        diagnostics={
            **pack.diagnostics,
            "compiled": True,
            "dropped_unrelated": dropped,
            "conflict_or_update_notes": len(conflict_notes),
            "arbitrated_updates": arbitration.get("updates", 0),
            "arbitrated_conflicts": arbitration.get("conflicts", 0),
            "preference_superseded": pref_superseded,
            "query_intent": query_intent(query),
            "question_layer": layer,
            "include_superseded": keep_superseded,
            "prompt_item_chars": 0,
        },
    )
    budget = token_budget or 1600
    _enforce_budget(context, budget)
    context.diagnostics["prompt_item_chars"] = _item_chars(context)
    return context


def _preference_tokens(items: list[ContextMemoryItem]) -> set[str]:
    tokens: set[str] = set()
    for item in items:
        tokens.update(query_content_tokens(item.text))
        for quoted in re.findall(r"「([^」]+)」", item.text):
            tokens.update(query_content_tokens(quoted))
            if quoted:
                tokens.add(quoted)
    tokens -= {"用户", "助手", "希望", "自称", "应该"}
    return tokens


def _lift_preference_structure(
    query: str,
    preference_hits: list[ContextMemoryItem],
    episodes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> tuple[list[ContextMemoryItem], list[ContextMemoryItem], list[ContextMemoryItem]]:
    episode_by_id = {str(item.get("episode_id") or ""): item for item in episodes}
    trajectory_by_id = {str(item.get("trajectory_id") or ""): item for item in trajectories}
    address_episode_ids: list[str] = []
    other_episode_ids: list[str] = []
    trajectory_ids: list[str] = []
    call_query = any(marker in query.lower() for marker in CALL_MARKERS)
    for item in preference_hits:
        metadata = item.metadata or {}
        preference_type = str(metadata.get("preference_type") or "")
        target = address_episode_ids if preference_type == "address" else other_episode_ids
        for episode_id in metadata.get("evidence_episode_ids") or []:
            episode_id = str(episode_id)
            if episode_id not in address_episode_ids and episode_id not in other_episode_ids:
                target.append(episode_id)
        for trajectory_id in metadata.get("evidence_trajectory_ids") or []:
            if trajectory_id not in trajectory_ids:
                trajectory_ids.append(str(trajectory_id))
    score = float(preference_hits[0].score if preference_hits else 0.6)
    background = [
        ContextMemoryItem(
            item_id=trajectory_id,
            text=_compact_payload_text(trajectory_by_id[trajectory_id], prefer_short=True),
            source="long_term_background",
            score=score,
            metadata={"lifted_from": "preference"},
        )
        for trajectory_id in trajectory_ids[:2]
        if trajectory_id in trajectory_by_id
    ]
    situation_ids = address_episode_ids[:2] if call_query else (address_episode_ids + other_episode_ids)[:2]
    related_ids = [item for item in other_episode_ids if item not in situation_ids][:2]
    if not call_query:
        related_ids = [item for item in address_episode_ids if item not in situation_ids][:2]
    situation = [
        _episode_from_payload(episode_id, episode_by_id[episode_id], "current_situation", score)
        for episode_id in situation_ids
        if episode_id in episode_by_id
    ]
    related = [
        _episode_from_payload(episode_id, episode_by_id[episode_id], "related_experience", score)
        for episode_id in related_ids
        if episode_id in episode_by_id
    ]
    return background, situation, related


def _episode_from_payload(
    episode_id: str,
    payload: dict[str, Any],
    source: str,
    score: float,
) -> ContextMemoryItem:
    return ContextMemoryItem(
        item_id=episode_id,
        text=_compact_payload_text(payload, prefer_short=True),
        source=source,
        score=score,
        timestamp=_optional_time(payload),
        metadata={
            "lifted_from": "preference",
            "status": item_status(payload),
            "cognitive_status": item_status(payload),
            "superseded_by": (payload.get("metadata") or {}).get("superseded_by"),
        },
    )


def _compact_payload_text(payload: dict[str, Any], *, prefer_short: bool = False) -> str:
    title = _clean_prompt_text(str(payload.get("title") or ""))
    summary = _clean_prompt_text(str(payload.get("summary") or payload.get("text") or ""))
    reason = _clean_prompt_text(str(payload.get("reason") or ""))
    body = _join_unique(title, summary)
    if prefer_short:
        short = title or _first_clause(body) or reason
        return _clip(short, 120)
    if reason and reason not in body and len(reason) <= 80:
        body = _join_unique(body, reason)
    return _clip(body, 180)


def _join_unique(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left == right or right.startswith(left) or left in right:
        return right if len(right) >= len(left) else left
    if right in left:
        return left
    return f"{left} {right}"


def _first_clause(text: str) -> str:
    compact = _clean_prompt_text(text)
    if not compact:
        return ""
    parts = re.split(r"[。！？!?\n]", compact, maxsplit=1)
    return (parts[0] or compact).strip()


def _clean_prompt_text(text: str) -> str:
    compact = TOOL_NOISE.sub(" ", str(text or ""))
    compact = re.sub(r"\s+", " ", compact).strip()
    compact = _dedupe_repeated_spans(compact)
    return compact


def _dedupe_repeated_spans(text: str) -> str:
    compact = text.strip()
    for size in range(len(compact) // 2, 7, -1):
        left = compact[:size].strip()
        right = compact[size:].strip()
        if left and left == right:
            compact = left
            break
    clauses = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s*", compact) if part.strip()]
    unique: list[str] = []
    for clause in clauses:
        if unique and (clause == unique[-1] or clause in unique[-1] or unique[-1] in clause):
            if len(clause) > len(unique[-1]):
                unique[-1] = clause
            continue
        unique.append(clause)
    return " ".join(unique).strip()


def _arbitrate_preferences(
    items: list[ContextMemoryItem],
    episodes: list[dict[str, Any]],
    *,
    keep_superseded: bool = False,
) -> tuple[list[ContextMemoryItem], int]:
    if len(items) <= 1:
        return items, 0
    episode_by_id = {
        str(item.get("episode_id") or ""): item
        for item in episodes
        if item.get("episode_id")
    }
    grouped: dict[str, list[ContextMemoryItem]] = {}
    for item in items:
        preference_type = str((item.metadata or {}).get("preference_type") or item.item_id)
        grouped.setdefault(preference_type, []).append(item)
    kept: list[ContextMemoryItem] = []
    superseded = 0
    for group in grouped.values():
        ranked = sorted(
            group,
            key=lambda item: _preference_rank(item, episode_by_id),
            reverse=True,
        )
        winners: list[ContextMemoryItem] = []
        losers: list[ContextMemoryItem] = []
        for item in ranked:
            if any(_preference_supersedes(existing, item) for existing in winners):
                superseded += 1
                losers.append(
                    _annotate(item, "superseded", "（已被后续更新）")
                )
                continue
            winners.append(item)
        kept.extend(winners)
        if keep_superseded:
            kept.extend(losers)
    kept.sort(key=lambda item: (-item.score, item.item_id))
    return kept, superseded


def _preference_supersedes(later: ContextMemoryItem, earlier: ContextMemoryItem) -> bool:
    later_type = str((later.metadata or {}).get("preference_type") or "")
    earlier_type = str((earlier.metadata or {}).get("preference_type") or "")
    if later_type != earlier_type:
        return False
    if later_type == "address":
        return True
    stop = {"用户", "助手", "希望", "应该", "自称", "户希", "望助"}
    later_tokens = query_content_tokens(_preference_body(later.text)) - stop
    earlier_tokens = query_content_tokens(_preference_body(earlier.text)) - stop
    return bool(later_tokens & earlier_tokens)


def _preference_body(text: str) -> str:
    return re.sub(r"^用户希望助手", "", str(text or "")).strip("。 ")


def _preference_rank(
    item: ContextMemoryItem,
    episode_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str, float]:
    times = []
    metadata = item.metadata or {}
    for episode_id in metadata.get("evidence_episode_ids") or []:
        payload = episode_by_id.get(str(episode_id)) or {}
        times.append(_optional_time(payload) or "")
    if item.timestamp:
        times.append(item.timestamp)
    evidence_ids = [str(value) for value in (metadata.get("evidence_ids") or [])]
    return (max(times) if times else "", evidence_ids[-1] if evidence_ids else "", item.score)


def _arbitrate_structure(
    situation: list[ContextMemoryItem],
    related: list[ContextMemoryItem],
    pack: HybridRecall,
    trajectories: list[dict[str, Any]],
) -> tuple[list[ContextMemoryItem], list[ContextMemoryItem], dict[str, int]]:
    superseded_by, conflict_ids = _relation_maps(pack, trajectories)
    inferred = _infer_supersession(situation + related)
    for earlier, later in inferred.items():
        superseded_by.setdefault(earlier, later)
    later_ids = set(superseded_by.values())
    new_situation: list[ContextMemoryItem] = []
    new_related: list[ContextMemoryItem] = []
    updates = 0
    conflicts = 0

    def place(item: ContextMemoryItem, *, from_situation: bool) -> None:
        nonlocal updates, conflicts
        later = superseded_by.get(item.item_id)
        if later or item_status(item.metadata) == "superseded":
            updates += 1
            new_related.append(
                _annotate(item, "superseded", "（已被后续更新）", source="related_experience")
            )
            return
        if item.item_id in conflict_ids:
            conflicts += 1
            target = new_situation if from_situation else new_related
            target.append(_annotate(item, "conflict", "（存在冲突，请并列保留）"))
            return
        if item.item_id in later_ids and from_situation:
            new_situation.append(
                _with_item(
                    item,
                    metadata={
                        "arbitration": "latest_update",
                        "status": "active",
                        "cognitive_status": "active",
                    },
                )
            )
            return
        if from_situation:
            new_situation.append(item)
        else:
            new_related.append(item)

    for item in situation:
        place(item, from_situation=True)
    for item in related:
        if item.item_id in later_ids and all(existing.item_id != item.item_id for existing in new_situation):
            new_situation.append(
                _with_item(
                    item,
                    source="current_situation",
                    metadata={
                        "arbitration": "latest_update",
                        "status": "active",
                        "cognitive_status": "active",
                    },
                )
            )
            updates += 1
            continue
        place(item, from_situation=False)

    new_situation = _dedupe_items(new_situation)
    new_related = [
        item
        for item in _dedupe_items(new_related)
        if item.item_id not in {existing.item_id for existing in new_situation}
    ]
    return new_situation, new_related, {"updates": updates, "conflicts": conflicts}


def _relation_maps(
    pack: HybridRecall,
    trajectories: list[dict[str, Any]],
) -> tuple[dict[str, str], set[str]]:
    superseded_by: dict[str, str] = {}
    conflict_ids: set[str] = set()
    notes = list(pack.relation_notes)
    for trajectory in trajectories:
        trajectory_id = str(trajectory.get("trajectory_id") or "")
        for relation in trajectory.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            notes.append({**relation, "trajectory_id": trajectory_id})
    for note in notes:
        relation_type = str(note.get("relation") or "").lower()
        source_id = str(note.get("source_episode_id") or "")
        target_id = str(note.get("target_episode_id") or "")
        if not source_id or not target_id or source_id == target_id:
            continue
        if relation_type in UPDATE_RELATIONS:
            superseded_by[source_id] = target_id
        elif relation_type == "conflict":
            conflict_ids.add(source_id)
            conflict_ids.add(target_id)
    return superseded_by, conflict_ids


def _infer_supersession(items: list[ContextMemoryItem]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ordered = sorted(items, key=lambda item: (item.timestamp or "", item.item_id))
    for index, earlier in enumerate(ordered):
        for later in ordered[index + 1 :]:
            if later.item_id == earlier.item_id:
                continue
            if not any(marker in later.text for marker in REVISION_MARKERS):
                continue
            later_tokens = query_content_tokens(later.text)
            earlier_tokens = query_content_tokens(earlier.text)
            if later_tokens & earlier_tokens or _overlaps(later.text, earlier_tokens):
                mapping[earlier.item_id] = later.item_id
    return mapping


def _annotate(
    item: ContextMemoryItem,
    role: str,
    prefix: str,
    *,
    source: str | None = None,
) -> ContextMemoryItem:
    text = item.text
    if prefix and prefix not in text:
        text = f"{prefix}{text}"
    extra = {"arbitration": role}
    if role == "superseded":
        extra["status"] = "superseded"
        extra["cognitive_status"] = "superseded"
    elif role in {"latest_update", "primary"}:
        extra["status"] = "active"
        extra["cognitive_status"] = "active"
    return _with_item(item, text=text, source=source, metadata=extra)


def _with_item(
    item: ContextMemoryItem,
    *,
    text: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextMemoryItem:
    merged = dict(item.metadata)
    if metadata:
        merged.update(metadata)
    return ContextMemoryItem(
        item_id=item.item_id,
        text=item.text if text is None else text,
        source=source or item.source,
        score=item.score,
        timestamp=item.timestamp,
        metadata=merged,
    )


def _dedupe_items(items: list[ContextMemoryItem]) -> list[ContextMemoryItem]:
    seen: set[str] = set()
    unique: list[ContextMemoryItem] = []
    for item in items:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        unique.append(item)
    return unique


def _optional_time(payload: dict[str, Any]) -> str | None:
    value = payload.get("start_time") or payload.get("end_time") or payload.get("timestamp")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _filter_slot(
    items: list[ContextMemoryItem],
    keep_tokens: set[str],
    dropped: int,
    *,
    require_overlap: bool,
) -> tuple[list[ContextMemoryItem], int]:
    if not require_overlap or not keep_tokens:
        return items, dropped
    kept: list[ContextMemoryItem] = []
    for item in items:
        if _overlaps(item.text, keep_tokens):
            kept.append(item)
        else:
            dropped += 1
    return kept, dropped


def _overlaps(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    folded = (text or "").lower()
    if any(token.lower() in folded for token in tokens if len(token) >= 2):
        return True
    return bool(query_content_tokens(text) & tokens)


def _drop_if_already_in(
    evidence: list[ContextMemoryItem],
    preferences: list[ContextMemoryItem],
) -> list[ContextMemoryItem]:
    pref_ids = {
        str(value)
        for item in preferences
        for value in (item.metadata or {}).get("evidence_ids") or []
    }
    return [item for item in evidence if item.item_id not in pref_ids]


def _merge_unique(
    primary: list[ContextMemoryItem],
    extra: list[ContextMemoryItem],
    dropped: int,
) -> tuple[list[ContextMemoryItem], int]:
    seen = {item.item_id for item in primary}
    merged = list(primary)
    for item in extra:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        merged.append(item)
    return merged, dropped


def _compress_slot(items: list[ContextMemoryItem], source: str) -> list[ContextMemoryItem]:
    limit = SLOT_LIMITS.get(source, 4)
    char_limit = ITEM_CHAR_LIMIT.get(source, 160)
    compressed: list[ContextMemoryItem] = []
    for item in items[:limit]:
        compressed.append(
            ContextMemoryItem(
                item_id=item.item_id,
                text=_clip(_clean_prompt_text(item.text), char_limit),
                source=item.source,
                score=item.score,
                timestamp=item.timestamp,
                metadata=dict(item.metadata),
            )
        )
    return compressed


def _clip(text: str, limit: int) -> str:
    compact = _clean_prompt_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _item_chars(context: CognitiveContext) -> int:
    total = 0
    for slot in (
        context.preference_memory,
        context.long_term_background,
        context.current_situation,
        context.related_experience,
        context.supporting_evidence,
        context.temporal_memory,
        context.execution_memory,
    ):
        total += sum(len(item.text) for item in slot)
    return total


def _enforce_budget(context: CognitiveContext, budget: int) -> None:
    if _item_chars(context) <= budget:
        return
    context.related_experience = context.related_experience[:1]
    context.temporal_memory = context.temporal_memory[:1]
    if _item_chars(context) <= budget:
        return
    context.long_term_background = context.long_term_background[:1]
    context.supporting_evidence = context.supporting_evidence[:2]
    if _item_chars(context) <= budget:
        return
    context.related_experience = []
    context.current_situation = context.current_situation[:1]
