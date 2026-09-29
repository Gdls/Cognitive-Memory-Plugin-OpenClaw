from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class LocalJsonStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.ensure_layout()

    def ensure_layout(self) -> None:
        for name in [
            "raw_events",
            "temporal",
            "memory_objects",
            "episodes",
            "trajectories",
            "preferences",
            "execution",
            "indexes",
            "traces",
            "checkpoints",
        ]:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def append_events(self, conversation_id: str, events: Iterable[dict[str, Any]]) -> None:
        rows = list(events)
        append_jsonl(self.root / "raw_events" / f"{conversation_id}.jsonl", rows)
        append_jsonl(self.root / "temporal" / f"{conversation_id}.jsonl", rows)

    def read_raw_events(self, conversation_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.root / "raw_events" / f"{conversation_id}.jsonl")

    def read_temporal_events(self, conversation_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.root / "temporal" / f"{conversation_id}.jsonl")

    def clear_temporal_events(self, conversation_id: str) -> None:
        path = self.root / "temporal" / f"{conversation_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def load_memory_objects(self, conversation_id: str) -> list[dict[str, Any]]:
        payload = read_json(self.root / "memory_objects" / f"{conversation_id}.json", default={"memories": []})
        return [item for item in payload.get("memories", []) if isinstance(item, dict)]

    def save_memory_objects(self, conversation_id: str, memories: list[dict[str, Any]]) -> None:
        write_json(self.root / "memory_objects" / f"{conversation_id}.json", {"memories": memories})

    def load_episodes(self, conversation_id: str) -> list[dict[str, Any]]:
        payload = read_json(self.root / "episodes" / f"{conversation_id}.json", default={"episodes": []})
        return [item for item in payload.get("episodes", []) if isinstance(item, dict)]

    def save_episodes(self, conversation_id: str, episodes: list[dict[str, Any]]) -> None:
        write_json(self.root / "episodes" / f"{conversation_id}.json", {"episodes": episodes})

    def load_trajectories(self, conversation_id: str) -> list[dict[str, Any]]:
        payload = read_json(self.root / "trajectories" / f"{conversation_id}.json", default={"trajectories": []})
        return [item for item in payload.get("trajectories", []) if isinstance(item, dict)]

    def save_trajectories(self, conversation_id: str, trajectories: list[dict[str, Any]]) -> None:
        write_json(self.root / "trajectories" / f"{conversation_id}.json", {"trajectories": trajectories})

    def load_preferences(self, conversation_id: str) -> list[dict[str, Any]]:
        payload = read_json(self.root / "preferences" / f"{conversation_id}.json", default={"preferences": []})
        return [item for item in payload.get("preferences", []) if isinstance(item, dict)]

    def save_preferences(self, conversation_id: str, preferences: list[dict[str, Any]]) -> None:
        write_json(self.root / "preferences" / f"{conversation_id}.json", {"preferences": preferences})

    def load_execution_memory(self, conversation_id: str) -> list[dict[str, Any]]:
        payload = read_json(self.root / "execution" / f"{conversation_id}.json", default={"execution": []})
        return [item for item in payload.get("execution", []) if isinstance(item, dict)]

    def save_execution_memory(self, conversation_id: str, items: list[dict[str, Any]]) -> None:
        write_json(self.root / "execution" / f"{conversation_id}.json", {"execution": items})

    def list_conversation_ids(self) -> list[str]:
        names: set[str] = set()
        for folder in ("memory_objects", "episodes", "raw_events", "temporal"):
            directory = self.root / folder
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.suffix in {".json", ".jsonl"} and path.stem:
                    names.add(path.stem)
        return sorted(names)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def read_json(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else dict(default)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
