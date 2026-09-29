"""MemOS-style progressive retrieval: excerpts first, then get / timeline.

Search returns short snippets plus ids. Full text and episode order are
follow-up tools. Auto-recall injects the same excerpt protocol, not the
compiled Cognitive Context dump.
"""

from __future__ import annotations

from typing import Any

from ..status import include_superseded_for_query, item_status

SLOT_KIND = {
    "supporting_evidence": "memory",
    "current_situation": "episode",
    "related_experience": "episode",
    "long_term_background": "trajectory",
    "preference_memory": "preference",
    "temporal_memory": "temporal",
    "execution_memory": "execution",
}

SLOT_ORDER = (
    "supporting_evidence",
    "current_situation",
    "related_experience",
    "long_term_background",
    "preference_memory",
    "temporal_memory",
)

LAYER_SLOT_ORDER = {
    "preference": (
        "preference_memory",
        "current_situation",
        "related_experience",
        "supporting_evidence",
        "long_term_background",
        "temporal_memory",
    ),
    "memory": SLOT_ORDER,
    "timeline": (
        "current_situation",
        "related_experience",
        "supporting_evidence",
        "long_term_background",
        "preference_memory",
        "temporal_memory",
    ),
    "trajectory": (
        "long_term_background",
        "current_situation",
        "related_experience",
        "supporting_evidence",
        "preference_memory",
        "temporal_memory",
    ),
}

PREFERENCE_MARKERS = (
    "怎么叫",
    "叫你",
    "叫我",
    "称呼",
    "人设",
    "偏好",
    "自称",
    "prefer",
    "what should i call",
    "call you",
    "call me",
    "风格",
    "以前叫",
    "曾经叫",
    "原来叫",
    "之前叫",
)
TIMELINE_MARKERS = (
    "过程",
    "经历",
    "顺序",
    "先后",
    "怎么回事",
    "时间线",
    "timeline",
    "一步步",
    "前后",
    "然后呢",
    "经过",
)
TRAJECTORY_MARKERS = (
    "课题",
    "背景",
    "主题",
    "长期",
    "项目背景",
    "一直在",
    "overview",
    "background",
    "整体",
)
MEMORY_MARKERS = (
    "原话",
    "具体",
    "细节",
    "哪一天",
    "哪句",
    "原文",
    "exactly",
    "verbatim",
    "详细",
    "原话是",
)

LAYER_PRIMARY_TOOL = {
    "preference": "cogmem_get",
    "memory": "cogmem_get",
    "timeline": "cogmem_timeline",
    "trajectory": "cogmem_get",
}

LAYER_HINT = {
    "preference": (
        "This looks like a preference or identity question. "
        "Use preference excerpts; call cogmem_get on a preference id if the name or style is unclear."
    ),
    "memory": (
        "This looks like a detail question. "
        "Use memory excerpts; call cogmem_get for the original wording."
    ),
    "timeline": (
        "This looks like a sequence or history question. "
        "Call cogmem_timeline on the episode_id to see ordered events."
    ),
    "trajectory": (
        "This looks like a background or topic question. "
        "Use trajectory excerpts; call cogmem_get on a trajectory id for the long-term theme."
    ),
}

EXCERPT_CHARS = 180
LONG_QUERY_CHARS = 80
MIN_EVIDENCE_FOR_SUFFICIENT = 1
MIN_HITS_FOR_SUFFICIENT = 2

SEARCH_HINT = (
    "CogMem auto-recall found no relevant memories for this turn.\n"
    "If the user refers to past conversations, preferences, or history, "
    "call cogmem_search with a short 2-5 word query you generate."
)


def question_layer(query: str) -> str:
    """Rule-based layer for progressive expansion. No LLM.

    preference → Preference Memory
    memory     → Memory (original wording)
    timeline   → Episode timeline
    trajectory → Trajectory background
    """
    lowered = (query or "").lower()
    if any(marker in lowered for marker in TIMELINE_MARKERS):
        return "timeline"
    if any(marker in lowered for marker in PREFERENCE_MARKERS):
        return "preference"
    if any(marker in lowered for marker in TRAJECTORY_MARKERS):
        return "trajectory"
    if any(marker in lowered for marker in MEMORY_MARKERS):
        return "memory"
    return "memory"


def slot_order_for(query: str) -> tuple[str, ...]:
    return LAYER_SLOT_ORDER.get(question_layer(query), SLOT_ORDER)


def compact_hits(
    context: dict[str, Any],
    *,
    limit: int = 12,
    excerpt_chars: int = EXCERPT_CHARS,
    query: str = "",
    include_superseded: bool | None = None,
) -> list[dict[str, Any]]:
    packed = context.get("cognitive_context") or {}
    conversation_id = str(context.get("conversation_id") or "")
    query_text = query or str(context.get("query") or packed.get("query") or "")
    layer = question_layer(query_text)
    keep_superseded = include_superseded_for_query(
        query_text, include_superseded=bool(include_superseded)
    )
    hits: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for slot in slot_order_for(query_text):
        for item in packed.get(slot) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            kind = _kind_of(item, slot)
            item_id = str(item.get("item_id") or "")
            episode_id = _episode_id(kind, item_id, metadata)
            status = item_status(item)
            row = {
                "id": item_id,
                "kind": kind,
                "slot": slot,
                "score": float(item.get("score") or 0),
                "snippet": _excerpt(text, excerpt_chars),
                "episode_id": episode_id,
                "conversation_id": conversation_id,
                "status": status,
                "layer": layer,
                "follow_up": follow_up_for(kind, episode_id, layer=layer),
            }
            if status == "superseded" and not keep_superseded:
                skipped.append(row)
            else:
                hits.append(row)
            if len(hits) >= limit:
                return hits
    if not hits:
        return skipped[:limit]
    return hits


def follow_up_for(kind: str, episode_id: str | None, *, layer: str = "memory") -> list[str]:
    tools: list[str] = []
    if layer == "timeline":
        if kind == "episode" or episode_id:
            tools.append("cogmem_timeline")
        if kind in {"memory", "episode", "trajectory", "preference"}:
            tools.append("cogmem_get")
        return list(dict.fromkeys(tools))
    if kind in {"memory", "episode", "trajectory", "preference"}:
        tools.append("cogmem_get")
    if layer == "memory" and (kind == "episode" or episode_id):
        tools.append("cogmem_timeline")
    return list(dict.fromkeys(tools))


def is_sufficient(hits: list[dict[str, Any]], query: str = "") -> bool:
    if len(str(query or "")) >= LONG_QUERY_CHARS:
        return False
    if not hits:
        return False
    layer = question_layer(query)
    kinds = {str(hit.get("kind") or "") for hit in hits}
    if layer == "timeline":
        return False
    if layer == "preference":
        return "preference" in kinds or len(hits) >= MIN_HITS_FOR_SUFFICIENT
    if layer == "trajectory":
        return "trajectory" in kinds
    evidence = [hit for hit in hits if hit.get("kind") in {"memory", "episode"}]
    return len(evidence) >= MIN_EVIDENCE_FOR_SUFFICIENT and len(hits) >= MIN_HITS_FOR_SUFFICIENT


def next_tools(hits: list[dict[str, Any]], *, sufficient: bool, query: str = "") -> list[str]:
    if not hits:
        return ["cogmem_search"]
    layer = question_layer(query)
    primary = LAYER_PRIMARY_TOOL.get(layer, "cogmem_get")
    available: list[str] = []
    for hit in hits:
        available.extend(hit.get("follow_up") or [])
    available = list(dict.fromkeys(available))
    tools: list[str] = []
    if primary in available:
        tools.append(primary)
    for name in available:
        if name not in tools:
            tools.append(name)
    if not sufficient:
        tools.append("cogmem_search")
    return list(dict.fromkeys(tools))


def format_progressive_prepend(
    hits: list[dict[str, Any]],
    *,
    sufficient: bool,
    suggested: list[str],
    query: str = "",
) -> str:
    if not hits:
        return SEARCH_HINT
    layer = question_layer(query)
    lines = [
        "## CogMem recalled excerpts (not full memory)",
        "",
        f"Question layer: {layer}.",
        "",
    ]
    for index, hit in enumerate(hits, start=1):
        episode = hit.get("episode_id") or ""
        episode_bit = f" episode_id={episode}" if episode else ""
        status = str(hit.get("status") or "")
        status_bit = f" status={status}" if status and status != "active" else ""
        score = float(hit.get("score") or 0)
        lines.append(
            f"{index}. [{hit.get('kind')}:{hit.get('id')}] {hit.get('snippet')} "
            f"(score={score:.3f}){episode_bit}{status_bit}"
        )
        follow = hit.get("follow_up") or []
        calls = _next_calls_for_hit(hit, follow, layer=layer, sufficient=sufficient)
        if calls:
            lines.append(f"   next: {' ; '.join(calls)}")
    lines.append("")
    lines.append(LAYER_HINT.get(layer, LAYER_HINT["memory"]))
    if sufficient:
        lines.append(
            "These are excerpts. Expand only if needed with the next tool for this question layer."
        )
    else:
        names = ", ".join(suggested) if suggested else "cogmem_get, cogmem_timeline, cogmem_search"
        lines.append(f"These excerpts may be insufficient. Next tools: {names}.")
        if layer == "timeline":
            lines.append("Prefer cogmem_timeline for the episode sequence.")
        elif layer == "trajectory":
            lines.append("Prefer cogmem_get on a trajectory id for the long-term theme.")
        elif layer == "preference":
            lines.append("Prefer cogmem_get on a preference id for the current name or style.")
        else:
            lines.append("Prefer cogmem_get for original text.")
        lines.append(
            "If the angle is wrong, call cogmem_search with a short 2-5 word query you generate."
        )
    return "\n".join(lines)


def progressive_recall_fields(
    context: dict[str, Any],
    *,
    query: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    query_text = query or str(context.get("query") or "")
    hits = compact_hits(context, limit=limit, query=query_text)
    sufficient = is_sufficient(hits, query_text)
    suggested = next_tools(hits, sufficient=sufficient, query=query_text)
    prepend = format_progressive_prepend(
        hits, sufficient=sufficient, suggested=suggested, query=query_text
    )
    layer = question_layer(query_text)
    return {
        "hits": hits,
        "sufficient": sufficient,
        "next_tools": suggested,
        "question_layer": layer,
        "openclaw": {
            "conversationId": context.get("conversation_id") or "",
            "prependContext": prepend,
            "source": "cogmem",
            "sufficient": sufficient,
            "nextTools": suggested,
            "questionLayer": layer,
        },
    }


def empty_progressive_recall(conversation_id: str = "", query: str = "") -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "query": query,
        "cognitive_context": {},
        "prompt_text": "",
        "hits": [],
        "sufficient": False,
        "next_tools": ["cogmem_search"],
        "question_layer": question_layer(query),
        "openclaw": {
            "conversationId": conversation_id,
            "prependContext": SEARCH_HINT,
            "source": "cogmem",
            "sufficient": False,
            "nextTools": ["cogmem_search"],
            "questionLayer": question_layer(query),
        },
    }


def _next_calls_for_hit(
    hit: dict[str, Any],
    follow: list[str],
    *,
    layer: str,
    sufficient: bool,
) -> list[str]:
    get_call = ""
    timeline_call = ""
    if "cogmem_get" in follow and hit.get("id"):
        get_call = f'cogmem_get(item_id="{hit["id"]}", kind="{hit.get("kind")}")'
    if "cogmem_timeline" in follow:
        episode_id = hit.get("episode_id") or (
            hit.get("id") if hit.get("kind") == "episode" else ""
        )
        if episode_id:
            timeline_call = f'cogmem_timeline(episode_id="{episode_id}")'
    if layer == "timeline":
        return [item for item in (timeline_call, get_call) if item]
    if sufficient:
        return [get_call] if get_call else []
    return [item for item in (get_call, timeline_call) if item]


def _kind_of(item: dict[str, Any], slot: str) -> str:
    source = str(item.get("source") or "")
    if source in SLOT_KIND:
        return SLOT_KIND[source]
    if slot in SLOT_KIND:
        return SLOT_KIND[slot]
    return source or slot or "memory"


def _episode_id(kind: str, item_id: str, metadata: dict[str, Any]) -> str:
    if kind == "episode":
        return item_id
    value = metadata.get("episode_id")
    if value:
        return str(value)
    episode_ids = metadata.get("episode_ids") or []
    if episode_ids:
        return str(episode_ids[0])
    return ""


def _excerpt(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
