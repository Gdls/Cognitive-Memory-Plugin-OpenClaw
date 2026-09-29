from __future__ import annotations

import json
import urllib.request
from typing import Any


class LlmHttpError(RuntimeError):
    pass


def chat_complete(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_seconds: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        text = str(body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmHttpError("LLM response missing choices[0].message.content") from exc
    if not text:
        raise LlmHttpError("LLM returned an empty response")
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("JSON is not an object", text, 0)
    return payload
