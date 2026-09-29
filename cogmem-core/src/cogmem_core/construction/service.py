from __future__ import annotations

from .builders import (
    build_episode_memories_from_objects,
    build_memory_objects_from_events,
    build_trajectory_memories_from_episodes,
    merge_unique_memories,
)
from .config import ConstructionConfig
from ..contracts.build import BuildRequest, BuildResult
from ..contracts.events import normalize_events
from ..preferences import derive_candidate_preferences
from ..repositories.protocols import MemoryRepository
from ..status import apply_cognitive_status, apply_preference_status


class ConstructionService:
    """Offline/async cognitive-memory construction boundary."""

    def __init__(
        self,
        repository: MemoryRepository,
        config: ConstructionConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or ConstructionConfig.from_env()

    def build(self, request: BuildRequest) -> BuildResult:
        conversation_id = request.conversation_id
        temporal_events = normalize_events(self.repository.read_temporal_events(conversation_id))
        raw_events = normalize_events(self.repository.read_raw_events(conversation_id))
        if request.force_rebuild:
            source_events = raw_events or temporal_events
            existing: list = []
        else:
            source_events = temporal_events
            existing = self.repository.load_memory_objects(conversation_id)
        new_memories = build_memory_objects_from_events(conversation_id, source_events)
        memories = merge_unique_memories(existing, new_memories)
        if not memories:
            return BuildResult(
                conversation_id=conversation_id,
                processed_events=len(source_events),
                new_memories=0,
                total_memories=0,
                episodes=len(self.repository.load_episodes(conversation_id)),
                trajectories=len(self.repository.load_trajectories(conversation_id)),
            )
        self.repository.save_memory_objects(conversation_id, memories)

        episodes = [
            episode.to_dict()
            for episode in build_episode_memories_from_objects(
                conversation_id, memories, config=self.config
            )
        ]
        self.repository.save_episodes(conversation_id, episodes)

        trajectories = [
            trajectory.to_dict()
            for trajectory in build_trajectory_memories_from_episodes(
                conversation_id, episodes, config=self.config
            )
        ]
        self.repository.save_trajectories(conversation_id, trajectories)
        apply_cognitive_status(episodes, trajectories, memories)
        self.repository.save_memory_objects(conversation_id, memories)
        self.repository.save_episodes(conversation_id, episodes)
        self.repository.clear_temporal_events(conversation_id)
        preference_result = self.derive_preferences(conversation_id)

        return BuildResult(
            conversation_id=conversation_id,
            processed_events=len(source_events),
            new_memories=len(new_memories),
            total_memories=len(memories),
            episodes=len(episodes),
            trajectories=len(trajectories),
            preferences=int(preference_result["total_preferences"]),
        )

    def derive_preferences(self, conversation_id: str) -> dict[str, object]:
        memories = self.repository.load_memory_objects(conversation_id)
        episodes = self.repository.load_episodes(conversation_id)
        trajectories = self.repository.load_trajectories(conversation_id)
        existing = self.repository.load_preferences(conversation_id)
        candidates = [
            item.to_dict()
            for item in derive_candidate_preferences(
                conversation_id,
                memories,
                episodes=episodes,
                trajectories=trajectories,
            )
        ]
        merged = merge_by_id(existing, candidates, "memory_id")
        apply_preference_status(merged, episodes, memories)
        self.repository.save_preferences(conversation_id, merged)
        return {
            "conversation_id": conversation_id,
            "candidate_preferences": len(candidates),
            "total_preferences": len(merged),
        }


def merge_by_id(
    existing: list[dict[str, object]],
    new_items: list[dict[str, object]],
    key: str,
) -> list[dict[str, object]]:
    merged = {str(item.get(key)): dict(item) for item in existing if item.get(key)}
    for item in new_items:
        if item.get(key):
            merged[str(item[key])] = dict(item)
    return list(merged.values())
