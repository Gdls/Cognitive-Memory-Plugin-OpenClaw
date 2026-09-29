"""Derive Preference Memory from user behavior, not as factual evidence.

Architecture: Preference Memory is behavior guidance (address, style, role,
constraints) compiled from repeated user turns and trajectory themes. It must
not be copied into Supporting Evidence.
"""

from __future__ import annotations

import re
from hashlib import sha1
from typing import Any

from .schemas import PreferenceMemory
from .status import include_superseded_for_query, item_status


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


EXPLICIT_MARKERS = ("prefer", "preference", "喜欢", "偏好", "希望", "不要")
ASSISTANT_SPEAKERS = {"assistant", "system", "tool", "bot"}
MAX_CONTENT_CHARS = 120

ADDRESS_BOT = re.compile(
    r"(?:你叫|你的名字(?:是|叫)?|自称)\s*[「\"']?([^，。,\s「」\"']{1,12})"
)
ADDRESS_USER_RELATION = re.compile(r"我是你的\s*([^，。,\s]{1,12})")
ADDRESS_USER_KIN = re.compile(r"你是(?:我的)?(姐夫|表哥|哥哥|姐姐|老板|主人|老师|朋友)")
ADDRESS_CALL_ME = re.compile(r"(?:叫我|称呼我|喊我)\s*([^，。,\s]{1,12})")
PREFER_ZH = re.compile(r"(?:我(?:比较|更)?(?:偏好|喜欢|希望)|偏好|喜欢)([^。\n]{2,40})")
PREFER_EN = re.compile(r"(?i)\b(?:i\s+)?prefer(?:s| to)?\s+([^.\n]{3,60})")
NO_ALWAYS = re.compile(r"不用总是([^，。,\n]{2,20})")
DONT = re.compile(r"(?:不要|别再|别总是)([^，。,\n]{2,24})")
ROLE_LIFE = re.compile(r"(生活情绪[^，。,\n]{0,16}|情绪调节[^，。,\n]{0,12})")
NO_WORK = re.compile(r"(?:不碰工作|不处理那些工作|不用去处理那些工作|不接项目)")
USE_ADDRESS = re.compile(r"(?:要有称呼|带称呼|关系亲近)")
CONCISE = re.compile(r"(?:先(?:给)?结论|简洁|concise|implementation-first)", re.I)


def derive_candidate_preferences(
    conversation_id: str,
    memory_payloads: list[dict[str, Any]],
    *,
    episodes: list[dict[str, Any]] | None = None,
    trajectories: list[dict[str, Any]] | None = None,
) -> list[PreferenceMemory]:
    grouped: dict[str, PreferenceMemory] = {}
    user_memories = [item for item in memory_payloads if _is_user_memory(item)]
    for memory in user_memories:
        for preference_type, content, confidence in _extract_from_text(str(memory.get("text") or "")):
            _merge_group(
                grouped,
                conversation_id,
                preference_type=preference_type,
                content=content,
                confidence=confidence,
                evidence_ids=[str(memory.get("memory_id") or "")],
            )

    episode_by_id = {
        str(item.get("episode_id") or ""): item
        for item in (episodes or [])
        if item.get("episode_id")
    }
    memory_to_episode = {}
    for episode in episode_by_id.values():
        for memory_id in episode.get("memory_ids") or []:
            memory_to_episode[str(memory_id)] = str(episode.get("episode_id") or "")

    for trajectory in trajectories or []:
        trajectory_id = str(trajectory.get("trajectory_id") or "")
        theme = compact_text(
            " ".join(
                part
                for part in [
                    str(trajectory.get("title") or ""),
                    str(trajectory.get("summary") or ""),
                ]
                if part
            )
        )
        extracted = _extract_from_text(theme)
        if not extracted:
            continue
        episode_ids = [str(item) for item in (trajectory.get("episode_ids") or [])]
        evidence_ids = [
            str(memory_id)
            for episode_id in episode_ids
            for memory_id in (episode_by_id.get(episode_id) or {}).get("memory_ids") or []
        ]
        for preference_type, content, confidence in extracted:
            _merge_group(
                grouped,
                conversation_id,
                preference_type=preference_type,
                content=content,
                confidence=min(0.7, confidence),
                evidence_ids=evidence_ids[:6],
                episode_ids=episode_ids,
                trajectory_ids=[trajectory_id] if trajectory_id else [],
            )

    for preference in grouped.values():
        metadata = dict(preference.metadata)
        episode_ids = sorted(
            {
                memory_to_episode[memory_id]
                for memory_id in preference.evidence_ids
                if memory_id in memory_to_episode
            }
        )
        if episode_ids:
            metadata["evidence_episode_ids"] = episode_ids
        preference.metadata = metadata
        if preference.confidence >= 0.75:
            preference.status = "active"
    return sorted(grouped.values(), key=lambda item: (-item.confidence, item.preference_type, item.memory_id))


def select_preference_items(
    payloads: list[dict[str, Any]],
    query: str,
    *,
    limit: int = 4,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """Always surface stable guidance; query overlap only reorders.

    Default recall skips superseded preferences unless the query is about
    history. Pass include_superseded=True to force the old records back.
    """
    if not payloads:
        return []
    keep_superseded = include_superseded_for_query(query, include_superseded=include_superseded)
    query_folded = (query or "").lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    skipped: list[tuple[float, dict[str, Any]]] = []
    for item in payloads:
        content = str(item.get("content") or "")
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        overlap = 2.0 if content.lower() in query_folded or _overlap(query_folded, content.lower()) else 0.0
        status = item_status(item)
        status_bonus = 0.2 if status == "active" else (-0.5 if status == "superseded" else 0.0)
        row = (overlap + confidence + status_bonus, item)
        if status == "superseded" and not keep_superseded:
            skipped.append(row)
            continue
        scored.append(row)
    if not scored:
        scored = skipped
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("memory_id") or "")))
    return [item for _, item in scored[:limit]]


def _extract_from_text(text: str) -> list[tuple[str, str, float]]:
    text = compact_text(text)
    if not text or len(text) > 800:
        return []
    found: list[tuple[str, str, float]] = []

    bot_name = _first(ADDRESS_BOT, text)
    user_relation = (
        _first(ADDRESS_USER_RELATION, text)
        or _first(ADDRESS_USER_KIN, text)
        or _first(ADDRESS_CALL_ME, text)
    )
    if bot_name or user_relation:
        parts = []
        if user_relation:
            parts.append(f"用户是「{user_relation}」")
        if bot_name:
            parts.append(f"助手应自称「{bot_name}」")
        found.append(("address", "，".join(parts) + "。", 0.86))

    if USE_ADDRESS.search(text):
        found.append(("style", "用户希望对话带称呼，关系亲近。", 0.78))

    prefer_zh = PREFER_ZH.search(text)
    if prefer_zh:
        clause = compact_text(prefer_zh.group(1)).strip("的是为了")
        if clause and not _is_noise_clause(clause):
            found.append(("style", f"用户偏好{clause}。", 0.88))
    prefer_en = PREFER_EN.search(text)
    if prefer_en:
        clause = compact_text(prefer_en.group(1))
        if clause:
            found.append(("style", f"The user prefers {clause}.", 0.86))

    if CONCISE.search(text) and not any(item[1].startswith("用户偏好先") for item in found):
        if "结论" in text or "细节" in text:
            found.append(("style", "用户偏好先给结论再给细节。", 0.84))
        elif re.search(r"concise", text, re.I):
            found.append(("style", "The user prefers concise answers.", 0.84))

    no_always = NO_ALWAYS.search(text)
    if no_always:
        found.append(("constraint", f"用户不希望助手总是{compact_text(no_always.group(1))}。", 0.82))
    dont = DONT.search(text)
    if dont:
        clause = compact_text(dont.group(1))
        if clause and not _is_noise_clause(clause):
            found.append(("constraint", f"用户不希望{clause}。", 0.72))

    life = ROLE_LIFE.search(text)
    if life or NO_WORK.search(text):
        bits = []
        if life:
            bits.append(compact_text(life.group(1)) or "生活情绪调节助手")
        if NO_WORK.search(text):
            bits.append("不处理工作项目")
        found.append(("role", "用户希望助手" + "，".join(bits) + "。", 0.84))

    return found


def _merge_group(
    grouped: dict[str, PreferenceMemory],
    conversation_id: str,
    *,
    preference_type: str,
    content: str,
    confidence: float,
    evidence_ids: list[str],
    episode_ids: list[str] | None = None,
    trajectory_ids: list[str] | None = None,
) -> None:
    content = compact_text(content)[:MAX_CONTENT_CHARS]
    if not content:
        return
    key = f"{preference_type}:{_normalize_key(content)}"
    evidence = [item for item in evidence_ids if item]
    if key in grouped:
        existing = grouped[key]
        existing.confidence = max(existing.confidence, confidence)
        existing.evidence_ids = list(dict.fromkeys([*existing.evidence_ids, *evidence]))
        metadata = dict(existing.metadata)
        if trajectory_ids:
            metadata["evidence_trajectory_ids"] = list(
                dict.fromkeys([*(metadata.get("evidence_trajectory_ids") or []), *trajectory_ids])
            )
        if episode_ids:
            metadata["evidence_episode_ids"] = list(
                dict.fromkeys([*(metadata.get("evidence_episode_ids") or []), *episode_ids])
            )
        existing.metadata = metadata
        return
    digest = sha1(f"{conversation_id}:{key}".encode("utf-8")).hexdigest()[:12]
    grouped[key] = PreferenceMemory(
        memory_id=f"pref_{digest}",
        conversation_id=conversation_id,
        content=content,
        preference_type=preference_type,
        confidence=confidence,
        evidence_ids=evidence,
        status="candidate",
        metadata={
            "source": "trajectory_heuristic",
            "evidence_trajectory_ids": list(trajectory_ids or []),
            "evidence_episode_ids": list(episode_ids or []),
        },
    )


def _is_user_memory(memory: dict[str, Any]) -> bool:
    metadata = memory.get("metadata") or {}
    event_type = str(metadata.get("event_type") or memory.get("event_type") or "")
    if event_type == "user_message":
        return True
    if event_type in {"assistant_message", "tool_call", "tool_observation", "system_event"}:
        return False
    speaker = str(metadata.get("speaker") or memory.get("speaker") or "").lower()
    if speaker in ASSISTANT_SPEAKERS:
        return False
    return bool(speaker)


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    value = compact_text(match.group(1))
    value = value.strip("「」\"'：: ")
    return value or None


def _is_noise_clause(clause: str) -> bool:
    noise = ("进一步", "告诉我", "如果您", "有其他问题", "帮助")
    return any(marker in clause for marker in noise) or len(clause) < 2


def _normalize_key(content: str) -> str:
    return re.sub(r"\s+", "", content.lower())


def _overlap(query: str, content: str) -> bool:
    if not query or not content:
        return False
    tokens = [token for token in re.findall(r"[\w\u4e00-\u9fff]{2,}", content) if token not in {"用户", "助手", "希望"}]
    return any(token in query for token in tokens[:8])
