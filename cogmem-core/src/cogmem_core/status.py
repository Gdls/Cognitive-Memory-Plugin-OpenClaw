"""Persist cognitive status on stored memory: active vs superseded.

Construction writes the status. Default recall hides superseded items except
on history-process queries. Compile still annotates updates in Related Experience.
"""

from __future__ import annotations

import re
from typing import Any

UPDATE_RELATIONS = {"update", "correction"}
REVISION_MARKERS = ("不要", "不用", "别再", "不再", "不碰", "改成", "纠正", "其实是")
IDENTITY_HISTORY_MARKERS = (
    "以前叫",
    "曾经叫",
    "原来叫",
    "之前叫",
    "当时叫",
    "used to call",
    "used to be called",
    "originally called",
)
HISTORY_MARKERS = (
    "why",
    "history",
    "为什么",
    "怎么回事",
    "过程",
    "经历",
    "以前",
    "曾经",
    "原来",
    "当时",
    "过去",
)
CALL_ME = re.compile(r"(?:叫我|称呼我|喊我)\s*([^，。,\s]{1,12})")
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_CANDIDATE = "candidate"


def query_intent(query: str) -> str:
    lowered = (query or "").lower()
    if any(term in lowered for term in IDENTITY_HISTORY_MARKERS):
        return "history_process"
    if any(term in lowered for term in ("conflict", "contradict", "冲突", "矛盾")):
        return "conflict_check"
    identity_markers = (
        "怎么叫",
        "叫你",
        "称呼",
        "人设",
        "who are you",
        "what should i call",
    )
    if any(term in lowered for term in identity_markers):
        return "current_state"
    if any(term in lowered for term in ("current", "latest", "now", "现在", "目前", "最新")):
        return "current_state"
    if any(term in lowered for term in HISTORY_MARKERS):
        return "history_process"
    return "evidence_lookup"


def is_history_query(query: str) -> bool:
    return query_intent(query) == "history_process"


def is_current_state_query(query: str) -> bool:
    return query_intent(query) == "current_state"


def include_superseded_for_query(query: str, *, include_superseded: bool = False) -> bool:
    return bool(include_superseded) or is_history_query(query)


def item_status(payload: dict[str, Any] | None) -> str:
    if not payload:
        return STATUS_ACTIVE
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for value in (
        payload.get("cognitive_status"),
        metadata.get("cognitive_status"),
        payload.get("status"),
        metadata.get("status"),
        metadata.get("arbitration"),
    ):
        status = str(value or "").strip().lower()
        if status == "latest_update":
            return STATUS_ACTIVE
        if status in {STATUS_SUPERSEDED, STATUS_ACTIVE, STATUS_CANDIDATE}:
            return status
    return STATUS_ACTIVE


def stamp_status(payload: dict[str, Any], status: str, *, superseded_by: str | None = None) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    metadata["cognitive_status"] = status
    if payload.get("preference_type") or str(payload.get("memory_id") or "").startswith("pref_"):
        payload["status"] = status
    elif "episode_id" in payload:
        metadata["status"] = status
    if status == STATUS_SUPERSEDED and superseded_by:
        metadata["superseded_by"] = superseded_by
    else:
        metadata.pop("superseded_by", None)
    payload["metadata"] = metadata
    return payload


def apply_cognitive_status(
    episodes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]] | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stamp episode/memory metadata.status from relations, revisions, and address updates."""
    superseded_by = _episode_supersession_map(episodes, trajectories or [], memories or [])
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        later = superseded_by.get(episode_id)
        stamp_status(
            episode,
            STATUS_SUPERSEDED if later else STATUS_ACTIVE,
            superseded_by=later,
        )

    memory_to_episode: dict[str, str] = {}
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        for memory_id in episode.get("memory_ids") or []:
            memory_to_episode[str(memory_id)] = episode_id
    for memory in memories or []:
        episode_id = memory_to_episode.get(str(memory.get("memory_id") or ""))
        later = superseded_by.get(episode_id or "")
        stamp_status(
            memory,
            STATUS_SUPERSEDED if later else STATUS_ACTIVE,
            superseded_by=later,
        )

    return {
        "superseded_episodes": len(superseded_by),
        "superseded_by": dict(superseded_by),
    }


def apply_preference_status(
    preferences: list[dict[str, Any]],
    episodes: list[dict[str, Any]] | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Within a preference type, later competing records stay active; earlier ones are superseded."""
    episode_list = episodes or []
    memory_list = memories or []
    episode_by_id = {
        str(item.get("episode_id") or ""): item
        for item in episode_list
        if item.get("episode_id")
    }
    memory_by_id = {
        str(item.get("memory_id") or ""): item
        for item in memory_list
        if item.get("memory_id")
    }
    latest_address = _latest_call_me_name(episode_list, memory_list)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in preferences:
        key = str(item.get("preference_type") or item.get("memory_id") or "")
        grouped.setdefault(key, []).append(item)

    superseded = 0
    for preference_type, group in grouped.items():
        if preference_type == "address" and latest_address:
            superseded += _stamp_address_group(group, latest_address)
            continue
        ranked = sorted(
            group,
            key=lambda item: _preference_rank(item, episode_by_id, memory_by_id),
            reverse=True,
        )
        winners: list[dict[str, Any]] = []
        for item in ranked:
            if any(_preference_supersedes(existing, item) for existing in winners):
                stamp_status(
                    item,
                    STATUS_SUPERSEDED,
                    superseded_by=str(winners[0].get("memory_id") or ""),
                )
                superseded += 1
                continue
            stamp_status(item, STATUS_ACTIVE)
            winners.append(item)
    return {"superseded_preferences": superseded}


def _episode_supersession_map(
    episodes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> dict[str, str]:
    superseded_by: dict[str, str] = {}
    by_id = {str(item.get("episode_id") or ""): item for item in episodes if item.get("episode_id")}
    memory_by_id = {
        str(item.get("memory_id") or ""): item for item in memories if item.get("memory_id")
    }

    for trajectory in trajectories:
        for relation in trajectory.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            relation_type = str(relation.get("relation") or "").lower()
            source_id = str(relation.get("source_episode_id") or "")
            target_id = str(relation.get("target_episode_id") or "")
            if relation_type not in UPDATE_RELATIONS:
                continue
            if source_id and target_id and source_id != target_id:
                superseded_by[source_id] = target_id

    original_order = _ordered_episodes(episodes)
    for trajectory in trajectories:
        member_ids = [str(value) for value in (trajectory.get("episode_ids") or [])]
        members = [by_id[item] for item in member_ids if item in by_id]
        _infer_revision_pairs(_ordered_episodes(members), memory_by_id, superseded_by)

    _infer_address_pairs(original_order, memory_by_id, superseded_by)
    return _break_supersession_cycles(
        superseded_by,
        [str(item.get("episode_id") or "") for item in original_order],
    )


def _infer_revision_pairs(
    ordered: list[dict[str, Any]],
    memory_by_id: dict[str, dict[str, Any]],
    superseded_by: dict[str, str],
) -> None:
    for index, earlier in enumerate(ordered):
        earlier_id = str(earlier.get("episode_id") or "")
        earlier_text = _episode_text(earlier, memory_by_id)
        earlier_tokens = set(_cjk_bigrams(earlier_text))
        for later in ordered[index + 1 :]:
            later_id = str(later.get("episode_id") or "")
            later_text = _episode_text(later, memory_by_id)
            if not any(marker in later_text for marker in REVISION_MARKERS):
                continue
            later_tokens = set(_cjk_bigrams(later_text))
            if (earlier_tokens & later_tokens) or (
                CALL_ME.search(earlier_text) and CALL_ME.search(later_text)
            ):
                superseded_by[earlier_id] = later_id


def _infer_address_pairs(
    ordered: list[dict[str, Any]],
    memory_by_id: dict[str, dict[str, Any]],
    superseded_by: dict[str, str],
) -> None:
    named: list[tuple[str, str]] = []
    for episode in ordered:
        episode_id = str(episode.get("episode_id") or "")
        matches = list(CALL_ME.finditer(_episode_text(episode, memory_by_id)))
        if not matches:
            continue
        name = (matches[-1].group(1) or "").strip("「」\"'：: ")
        if name:
            named.append((episode_id, name))
    if len(named) < 2:
        return
    latest_id, latest_name = named[-1]
    for earlier_id, earlier_name in named[:-1]:
        if earlier_name != latest_name:
            superseded_by[earlier_id] = latest_id


def _ordered_episodes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = [
        (position, item)
        for position, item in enumerate(episodes)
        if item.get("episode_id")
    ]
    indexed.sort(
        key=lambda pair: (
            str(pair[1].get("end_time") or pair[1].get("start_time") or ""),
            pair[0],
        )
    )
    return [item for _, item in indexed]


def _break_supersession_cycles(superseded_by: dict[str, str], ordered_ids: list[str]) -> dict[str, str]:
    index = {episode_id: position for position, episode_id in enumerate(ordered_ids)}
    result: dict[str, str] = {}
    for source_id, target_id in superseded_by.items():
        if not source_id or not target_id or source_id == target_id:
            continue
        if target_id in index and source_id in index and index[target_id] < index[source_id]:
            source_id, target_id = target_id, source_id
        result[source_id] = target_id
    for source_id, target_id in list(result.items()):
        if result.get(target_id) == source_id:
            if index.get(source_id, 0) <= index.get(target_id, 0):
                result.pop(target_id, None)
            else:
                result.pop(source_id, None)
    return result


def _latest_call_me_name(
    episodes: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> str:
    memory_by_id = {
        str(item.get("memory_id") or ""): item
        for item in memories
        if item.get("memory_id")
    }
    latest = ""
    ordered_memories = sorted(
        memories,
        key=lambda item: (str(item.get("timestamp") or ""), str(item.get("memory_id") or "")),
    )
    scan_texts = [
        _episode_text(episode, memory_by_id) for episode in _ordered_episodes(episodes)
    ] + [str(item.get("text") or "") for item in ordered_memories]
    for text in scan_texts:
        for match in CALL_ME.finditer(text):
            name = (match.group(1) or "").strip("「」\"'：: ")
            if name:
                latest = name
    return latest


def _address_name_from_content(content: str) -> str:
    match = re.search(r"用户是「([^」]+)」", str(content or ""))
    if match:
        return (match.group(1) or "").strip()
    match = CALL_ME.search(str(content or ""))
    return ((match.group(1) if match else "") or "").strip("「」\"'：: ")


def _stamp_address_group(group: list[dict[str, Any]], latest_name: str) -> int:
    winner_id = ""
    superseded = 0
    for item in group:
        name = _address_name_from_content(str(item.get("content") or ""))
        if name == latest_name:
            stamp_status(item, STATUS_ACTIVE)
            winner_id = str(item.get("memory_id") or winner_id)
        else:
            stamp_status(item, STATUS_SUPERSEDED, superseded_by=winner_id or latest_name)
            superseded += 1
    if not winner_id:
        return superseded
    for item in group:
        if item_status(item) == STATUS_SUPERSEDED:
            stamp_status(
                item,
                STATUS_SUPERSEDED,
                superseded_by=winner_id,
            )
    return superseded


def _episode_text(episode: dict[str, Any], memory_by_id: dict[str, dict[str, Any]]) -> str:
    parts = [
        str(episode.get("title") or ""),
        str(episode.get("summary") or ""),
        str(episode.get("reason") or ""),
        str(episode.get("topic") or ""),
    ]
    for memory_id in episode.get("memory_ids") or []:
        memory = memory_by_id.get(str(memory_id)) or {}
        parts.append(str(memory.get("text") or ""))
    for memory in episode.get("memories") or []:
        if isinstance(memory, dict):
            parts.append(str(memory.get("text") or ""))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _cjk_bigrams(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[index : index + 2] for index in range(len(compact) - 1)]


def _preference_rank(
    item: dict[str, Any],
    episode_by_id: dict[str, dict[str, Any]],
    memory_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str, float]:
    times: list[str] = []
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for memory_id in item.get("evidence_ids") or []:
        payload = (memory_by_id or {}).get(str(memory_id)) or {}
        times.append(str(payload.get("timestamp") or ""))
    for episode_id in metadata.get("evidence_episode_ids") or []:
        payload = episode_by_id.get(str(episode_id)) or {}
        times.append(str(payload.get("end_time") or payload.get("start_time") or ""))
    evidence_ids = [str(value) for value in (item.get("evidence_ids") or [])]
    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    nonempty = [value for value in times if value]
    return (max(nonempty) if nonempty else "", evidence_ids[-1] if evidence_ids else "", confidence)


def _preference_supersedes(later: dict[str, Any], earlier: dict[str, Any]) -> bool:
    later_type = str(later.get("preference_type") or "")
    earlier_type = str(earlier.get("preference_type") or "")
    if later_type != earlier_type:
        return False
    if later_type == "address":
        return True
    later_text = str(later.get("content") or "")
    earlier_text = str(earlier.get("content") or "")
    return bool(set(_cjk_bigrams(later_text)) & set(_cjk_bigrams(earlier_text)))
