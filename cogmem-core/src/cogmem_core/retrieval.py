from __future__ import annotations

import math
import re
from typing import Any

from .schemas import ContextMemoryItem


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
QUERY_STOP_GRAMS = {
    "什么",
    "怎么",
    "怎样",
    "之前",
    "以后",
    "用户",
    "请问",
    "告诉",
    "一下",
    "这个",
    "那个",
    "可以",
    "帮我",
    "问过",
    "过什",
    "么天",
    "户之",
    "前问",
}


def rank_items(query: str, items: list[ContextMemoryItem], *, limit: int = 8) -> list[ContextMemoryItem]:
    query_tokens = query_content_tokens(query)
    if not query_tokens:
        return items[:limit]

    tokenized: list[tuple[set[str], str]] = []
    document_frequency = {token: 0 for token in query_tokens}
    for item in items:
        tokens = tokenize(item.text)
        text_folded = item.text.lower()
        tokenized.append((tokens, text_folded))
        for token in query_tokens:
            if token in tokens or token in text_folded:
                document_frequency[token] += 1

    document_count = max(len(items), 1)
    query_folded = query.lower()
    scored: list[ContextMemoryItem] = []
    for item, (tokens, text_folded) in zip(items, tokenized):
        score = 0.0
        if query_folded in text_folded:
            score += 8.0
        for token in query_tokens:
            if token not in tokens and token not in text_folded:
                continue
            idf = math.log((document_count + 1) / (document_frequency[token] + 0.5))
            score += max(idf, 0.01) * (3.0 if len(token) >= 2 else 0.2)
        if score <= 0:
            continue
        scored.append(
            ContextMemoryItem(
                item_id=item.item_id,
                text=item.text,
                source=item.source,
                score=score,
                timestamp=item.timestamp,
                metadata=dict(item.metadata),
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.item_id))[:limit]


def query_content_tokens(query: str) -> set[str]:
    tokens = {token for token in tokenize(query) if len(token) >= 2 and token not in QUERY_STOP_GRAMS}
    return tokens or {token for token in tokenize(query) if len(token) >= 2}


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_PATTERN.findall(text or ""):
        token = token.lower().strip()
        if not token:
            continue
        if CJK_PATTERN.fullmatch(token):
            tokens.update(token[index : index + 2] for index in range(len(token) - 1))
        elif CJK_PATTERN.search(token):
            for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", token):
                if CJK_PATTERN.fullmatch(seg):
                    tokens.update(seg[i : i + 2] for i in range(len(seg) - 1))
                elif seg:
                    tokens.add(seg)
        else:
            tokens.add(token)
    return tokens


def context_item_from_payload(payload: dict[str, Any], *, source: str) -> ContextMemoryItem:
    text = context_text_from_payload(payload)
    return ContextMemoryItem(
        item_id=str(
            payload.get("memory_id")
            or payload.get("episode_id")
            or payload.get("trajectory_id")
            or payload.get("event_id")
            or payload.get("id")
            or ""
        ),
        text=text,
        source=source,
        timestamp=optional_string(payload.get("timestamp")),
        metadata=dict(payload.get("metadata") or {}),
    )


def rank_payloads(
    query: str,
    payloads: list[dict[str, Any]],
    *,
    source: str,
    limit: int = 8,
) -> list[ContextMemoryItem]:
    return rank_items(
        query,
        [context_item_from_payload(payload, source=source) for payload in payloads],
        limit=limit,
    )


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def context_text_from_payload(payload: dict[str, Any]) -> str:
    direct_text = str(payload.get("text") or payload.get("content") or "")
    if direct_text:
        return direct_text
    parts = [
        str(payload.get("title") or ""),
        str(payload.get("summary") or ""),
    ]
    return " ".join(part for part in parts if part).strip()
