from __future__ import annotations

from pathlib import Path
from typing import Any

from .construction import ConstructionService
from .construction.config import ConstructionConfig
from .contracts.build import BuildRequest
from .contracts.events import CogMemEvent, normalize_events
from .contracts.recall import RecallRequest
from .execution import extract_execution_memory
from .recall import RecallService
from .repositories import LocalJsonStore, MemoryRepository

ITEM_KIND_LOADERS = {
    "memory": ("load_memory_objects", "memory_id"),
    "episode": ("load_episodes", "episode_id"),
    "trajectory": ("load_trajectories", "trajectory_id"),
    "preference": ("load_preferences", "memory_id"),
}


class CogMem:
    """Stable facade over log ingestion, offline construction, and online recall."""

    def __init__(
        self,
        store_path: str | Path = "./cogmem_store",
        *,
        enable_execution_memory: bool = False,
        repository: MemoryRepository | None = None,
        construction_config: ConstructionConfig | None = None,
    ) -> None:
        self.store = repository or LocalJsonStore(store_path)
        self.enable_execution_memory = enable_execution_memory
        self.construction_config = construction_config or ConstructionConfig.from_env()
        self.construction = ConstructionService(self.store, config=self.construction_config)
        self.recall = RecallService(self.store)

    def append_events(
        self,
        conversation_id: str,
        events: list[dict[str, Any] | CogMemEvent],
    ) -> dict[str, Any]:
        normalized = normalize_events(events)
        self.store.append_events(conversation_id, [event.to_dict() for event in normalized])
        if self.enable_execution_memory:
            existing = self.store.load_execution_memory(conversation_id)
            new_items = [item.to_dict() for item in extract_execution_memory(conversation_id, normalized)]
            self.store.save_execution_memory(
                conversation_id, merge_by_id(existing, new_items, "memory_id")
            )
        return {
            "conversation_id": conversation_id,
            "appended": len(normalized),
            "temporal_count": len(self.store.read_temporal_events(conversation_id)),
        }

    def build_memory(
        self,
        conversation_id: str | None = None,
        *,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        if conversation_id is None:
            raise ValueError("conversation_id is required for v0.1 local build")
        return self.construction.build(
            BuildRequest(conversation_id=conversation_id, force_rebuild=force_rebuild)
        ).to_dict()

    def derive_preferences(self, conversation_id: str | None = None) -> dict[str, Any]:
        if conversation_id is None:
            raise ValueError("conversation_id is required for v0.1 preference derivation")
        return dict(self.construction.derive_preferences(conversation_id))

    def retrieve_context(
        self,
        conversation_id: str,
        query: str,
        *,
        include_temporal: bool = True,
        include_preferences: bool = True,
        include_execution: bool = False,
        include_superseded: bool = False,
        limit: int = 8,
    ) -> dict[str, Any]:
        request = RecallRequest(
            conversation_id=conversation_id,
            query=query,
            include_temporal=include_temporal,
            include_preferences=include_preferences,
            include_execution=include_execution,
            include_superseded=include_superseded,
            limit=limit,
        )
        return self.recall.recall(request).to_dict()

    def format_prompt_context(self, conversation_id: str, query: str, **kwargs: Any) -> str:
        return str(self.retrieve_context(conversation_id, query, **kwargs)["prompt_text"])

    def get_status(self, conversation_id: str | None = None) -> dict[str, Any]:
        if conversation_id is None:
            root = getattr(self.store, "root", None)
            list_ids = getattr(self.store, "list_conversation_ids", None)
            conversation_ids = list(list_ids()) if callable(list_ids) else []
            return {
                "store_path": str(root) if root is not None else None,
                "conversation_ids": conversation_ids,
                "conversation_count": len(conversation_ids),
            }
        return {
            "conversation_id": conversation_id,
            "raw_events": len(self.store.read_raw_events(conversation_id)),
            "temporal_events": len(self.store.read_temporal_events(conversation_id)),
            "memory_objects": len(self.store.load_memory_objects(conversation_id)),
            "episodes": len(self.store.load_episodes(conversation_id)),
            "trajectories": len(self.store.load_trajectories(conversation_id)),
            "preferences": len(self.store.load_preferences(conversation_id)),
            "execution_memory": len(self.store.load_execution_memory(conversation_id)),
        }

    def get_item(
        self,
        item_id: str,
        *,
        conversation_id: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Lookup one memory / episode / trajectory / preference by id."""
        item_id = str(item_id or "").strip()
        if not item_id:
            return {"found": False, "kind": kind, "id": item_id, "item": None}

        if kind and kind not in ITEM_KIND_LOADERS:
            return {"found": False, "kind": kind, "id": item_id, "item": None}

        kinds = [kind] if kind else list(ITEM_KIND_LOADERS)
        for candidate_id in self._conversation_ids(conversation_id):
            for candidate_kind in kinds:
                loader_name, id_key = ITEM_KIND_LOADERS[candidate_kind]
                loader = getattr(self.store, loader_name)
                for item in loader(candidate_id):
                    if str(item.get(id_key) or "") != item_id:
                        continue
                    return {
                        "found": True,
                        "kind": candidate_kind,
                        "conversation_id": candidate_id,
                        "id": item_id,
                        "item": dict(item),
                    }
        return {"found": False, "kind": kind, "id": item_id, "item": None}

    def get_timeline(
        self,
        episode_id: str,
        *,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return the ordered memory objects inside one episode."""
        episode_id = str(episode_id or "").strip()
        cap = max(1, min(int(limit or 20), 100))
        if not episode_id:
            return {"found": False, "episode_id": episode_id, "episode": None, "memories": []}

        for candidate_id in self._conversation_ids(conversation_id):
            episode = next(
                (
                    item
                    for item in self.store.load_episodes(candidate_id)
                    if str(item.get("episode_id") or "") == episode_id
                ),
                None,
            )
            if episode is None:
                continue
            memory_by_id = {
                str(item.get("memory_id") or ""): item
                for item in self.store.load_memory_objects(candidate_id)
                if item.get("memory_id")
            }
            memories = [
                dict(memory_by_id[memory_id])
                for memory_id in (episode.get("memory_ids") or [])
                if str(memory_id) in memory_by_id
            ][:cap]
            return {
                "found": True,
                "conversation_id": candidate_id,
                "episode_id": episode_id,
                "episode": dict(episode),
                "memories": memories,
            }
        return {"found": False, "episode_id": episode_id, "episode": None, "memories": []}

    def _conversation_ids(self, conversation_id: str | None) -> list[str]:
        if conversation_id:
            return [conversation_id]
        list_ids = getattr(self.store, "list_conversation_ids", None)
        if callable(list_ids):
            return list(list_ids())
        return []


def merge_by_id(
    existing: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    merged = {str(item.get(key)): dict(item) for item in existing if item.get(key)}
    for item in new_items:
        if item.get(key):
            merged[str(item[key])] = dict(item)
    return list(merged.values())
