from __future__ import annotations

import re
from typing import Any

TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "with", "you",
}


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def memory_text(item: dict[str, Any]) -> str:
    if item.get("text"):
        return compact_text(str(item["text"]))
    memory = item.get("memory") or {}
    semantic = memory.get("semantic") or {}
    return compact_text(str(semantic.get("raw_text") or semantic.get("summary") or ""))


def memory_speaker(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    if metadata.get("speaker"):
        return str(metadata["speaker"])
    locomo = item.get("locomo") or {}
    return str(locomo.get("speaker") or "unknown")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text or ""):
        token = token.lower().strip()
        if not token:
            continue
        if CJK_PATTERN.fullmatch(token):
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
            continue
        if len(token) > 1 and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
