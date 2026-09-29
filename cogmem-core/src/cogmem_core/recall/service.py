from __future__ import annotations

from .compile import compile_cognitive_context, is_guidance_query, text_overlaps_tokens
from ..retrieval import query_content_tokens
from .formatting import format_cognitive_context
from .ranking import context_item_from_payload, rank_items
from ..contracts.recall import ContextMemoryItem, ContextPack, RecallRequest
from ..preferences import select_preference_items
from ..repositories.protocols import MemoryRepository
from .bidirectional import hybrid_recall
from ..status import include_superseded_for_query, item_status


class RecallService:
    """Low-latency online memory retrieval and context assembly boundary."""

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def recall(self, request: RecallRequest) -> ContextPack:
        conversation_id = request.conversation_id
        query = request.query
        memory_payloads = self.repository.load_memory_objects(conversation_id)
        episodes = self.repository.load_episodes(conversation_id)
        trajectories = self.repository.load_trajectories(conversation_id)
        memories = [
            context_item_from_payload(item, source="long_term_background")
            for item in memory_payloads
        ]
        preference_payloads = self.repository.load_preferences(conversation_id)
        temporal = [
            context_item_from_payload(item, source="temporal_memory")
            for item in self.repository.read_temporal_events(conversation_id)
        ]
        execution = [
            ContextMemoryItem(
                item_id=str(item.get("memory_id") or ""),
                text=str(item.get("result_summary") or ""),
                source="execution_memory",
                timestamp=item.get("timestamp"),
                metadata=dict(item),
            )
            for item in self.repository.load_execution_memory(conversation_id)
        ]

        limit = request.limit
        temporal_hits = rank_items(query, temporal, limit=limit) if request.include_temporal else []
        preference_hits = []
        if request.include_preferences:
            selected = select_preference_items(
                preference_payloads,
                query,
                limit=min(4, limit),
                include_superseded=request.include_superseded,
            )
            preference_hits = [
                ContextMemoryItem(
                    item_id=str(item.get("memory_id") or ""),
                    text=str(item.get("content") or ""),
                    source="preference_memory",
                    score=float(item.get("confidence") or 0.0),
                    metadata={
                        "preference_type": item.get("preference_type"),
                        "status": item.get("status"),
                        "cognitive_status": (item.get("metadata") or {}).get("cognitive_status")
                        or item.get("status"),
                        "superseded_by": (item.get("metadata") or {}).get("superseded_by"),
                        "evidence_ids": item.get("evidence_ids") or [],
                        "evidence_episode_ids": (item.get("metadata") or {}).get("evidence_episode_ids") or [],
                        "evidence_trajectory_ids": (item.get("metadata") or {}).get("evidence_trajectory_ids") or [],
                    },
                )
                for item in selected
            ]
        execution_hits = rank_items(query, execution, limit=limit) if request.include_execution else []

        pack = hybrid_recall(
            query,
            memories=memory_payloads,
            episodes=episodes,
            trajectories=trajectories,
            top_trajectories=min(3, max(1, limit)),
            top_episodes=min(5, max(1, limit)),
            max_snippets=limit,
            snippets_per_episode=3,
        )
        keyword_hits = rank_items(query, memories, limit=limit)
        if not include_superseded_for_query(query, include_superseded=request.include_superseded):
            current_hits = [
                item for item in keyword_hits if item_status(item.metadata) != "superseded"
            ]
            if current_hits:
                keyword_hits = current_hits
        context = compile_cognitive_context(
            query=query,
            conversation_id=conversation_id,
            pack=pack,
            preference_hits=preference_hits,
            temporal_hits=temporal_hits,
            execution_hits=execution_hits,
            episodes=episodes,
            trajectories=trajectories,
            token_budget=request.token_budget,
            include_superseded=request.include_superseded,
        )
        if not context.supporting_evidence and keyword_hits and not is_guidance_query(query):
            keep_tokens = query_content_tokens(query)
            filled = [
                item
                for item in keyword_hits
                if item.score > 0 and text_overlaps_tokens(item.text, keep_tokens)
            ]
            context.supporting_evidence = filled[: min(4, limit)]
        if not context.current_situation and not context.preference_memory:
            context.current_situation = (context.long_term_background[:2] or temporal_hits[:2])
        context.diagnostics.update(
            {
                "long_term_candidates": len(memories),
                "episode_candidates": len(episodes),
                "trajectory_candidates": len(trajectories),
                "temporal_candidates": len(temporal),
                "preference_candidates": len(preference_payloads),
                "execution_candidates": len(execution),
                "relation_note_count": len(pack.relation_notes),
            }
        )
        return ContextPack(
            conversation_id=conversation_id,
            query=query,
            cognitive_context=context,
            prompt_text=format_cognitive_context(context),
        )
