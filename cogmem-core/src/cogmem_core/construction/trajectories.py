from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..contracts.memory import TrajectoryMemory
from .bm25 import BM25Index, retrieve_bm25_candidates
from .config import ConstructionConfig
from .llm import LlmHttpError, chat_complete, extract_json_object
from .text import compact_text, jaccard, memory_text


@dataclass(slots=True)
class EpisodeRelationDecision:
    related: bool
    relation: str
    trajectory_hint: str
    confidence: float
    reason: str
    source: str


@dataclass(slots=True)
class TitleReasonClusterConfig:
    top_k: int = 12
    temporal_next_k: int = 4
    min_bm25_score: float = 0.0
    bm25_weight: float = 0.7
    temporal_weight: float = 0.3
    assignment_threshold: float = 0.75
    key_episode_count: int = 5
    include_singletons: bool = True
    expansion_mode: str = "fixed_seed"


class TitleReasonClusterClassifier(Protocol):
    def classify_title_reason_profile(
        self,
        trajectory_episodes: list[dict[str, Any]],
        target_episode: dict[str, Any],
        *,
        key_episode_count: int,
    ) -> EpisodeRelationDecision:
        ...


class LexicalClusterClassifier:
    """Non-LLM membership judge for the LoCoMo BM25 cluster loop."""

    def __init__(self, *, related_threshold: float = 0.12) -> None:
        self.related_threshold = related_threshold

    def classify_title_reason_profile(
        self,
        trajectory_episodes: list[dict[str, Any]],
        target_episode: dict[str, Any],
        *,
        key_episode_count: int,
    ) -> EpisodeRelationDecision:
        query = cluster_query_text(trajectory_episodes, key_episode_count=key_episode_count)
        target = episode_title_reason_text(target_episode)
        overlap = jaccard(query, target)
        related = overlap >= self.related_threshold
        revision_markers = ("不要", "不用", "别再", "不再", "不碰", "改成", "纠正", "其实是")
        if related and any(marker in target for marker in revision_markers):
            relation = "correction" if any(marker in target for marker in ("纠正", "其实是")) else "update"
        elif overlap >= 0.35:
            relation = "continuation"
        elif related:
            relation = "progression"
        else:
            relation = "unrelated"
        # Map overlap onto the LoCoMo assignment threshold scale (~0.75).
        confidence = min(1.0, 0.4 + overlap * 1.6) if related else overlap
        return EpisodeRelationDecision(
            related=related,
            relation=relation,
            trajectory_hint=compact_text(str(trajectory_episodes[0].get("title") or ""))[:80],
            confidence=confidence,
            reason=f"lexical jaccard={overlap:.3f}",
            source="lexical",
        )


class OpenAICompatibleClusterClassifier:
    def __init__(self, config: ConstructionConfig) -> None:
        if not config.llm_ready:
            raise ValueError("locomo_llm trajectory builder requires LLM credentials")
        self.config = config

    def classify_title_reason_profile(
        self,
        trajectory_episodes: list[dict[str, Any]],
        target_episode: dict[str, Any],
        *,
        key_episode_count: int,
    ) -> EpisodeRelationDecision:
        prompt = (
            f"Continuity mode: {self.config.continuity_mode}\n"
            "Decide whether the candidate episode belongs to the same trajectory as the accepted cluster.\n"
            "Return JSON only: "
            '{"related": true, "relation": "continuation|progression|update|correction|conflict|unrelated",'
            ' "confidence": 0.8, "trajectory_hint": "short hint", "reason": "why"}\n\n'
            f"Cluster:\n{cluster_query_text(trajectory_episodes, key_episode_count=key_episode_count)}\n\n"
            f"Candidate:\n{episode_title_reason_text(target_episode)}\n"
        )
        text = chat_complete(
            base_url=str(self.config.llm_base_url),
            api_key=str(self.config.llm_api_key),
            model=str(self.config.llm_model),
            prompt=prompt,
            timeout_seconds=self.config.llm_timeout_seconds,
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = extract_json_object(text)
        relation = str(payload.get("relation") or "unrelated")
        related = bool(payload.get("related")) and relation != "unrelated"
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return EpisodeRelationDecision(
            related=related,
            relation=relation,
            trajectory_hint=str(payload.get("trajectory_hint") or "")[:80],
            confidence=max(0.0, min(1.0, confidence)),
            reason=str(payload.get("reason") or ""),
            source="llm",
        )


def build_title_reason_clusters(
    episodes: list[dict[str, Any]],
    classifier: TitleReasonClusterClassifier,
    *,
    sample_id: str,
    config: TitleReasonClusterConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or TitleReasonClusterConfig()
    representations = [episode_title_reason_text(episode) for episode in episodes]
    bm25 = BM25Index(representations)
    trajectories: list[dict[str, Any]] = []
    unassigned = set(range(len(episodes)))

    while unassigned:
        seed_index = min(unassigned)
        accepted_indices = [seed_index]
        unassigned.remove(seed_index)
        cluster_checks: list[dict[str, Any]] = []
        fixed_candidates = retrieve_bm25_candidates(
            bm25,
            episode_title_reason_text(episodes[seed_index]),
            {index for index in unassigned if index > seed_index},
            reference_indices=[seed_index],
            top_k=config.top_k,
            temporal_next_k=config.temporal_next_k,
            min_bm25_score=config.min_bm25_score,
            bm25_weight=config.bm25_weight,
            temporal_weight=config.temporal_weight,
        )
        last_accepted_index = seed_index
        for candidate in fixed_candidates:
            if candidate.episode_index not in unassigned:
                continue
            if candidate.episode_index <= last_accepted_index:
                continue
            decision = classifier.classify_title_reason_profile(
                [episodes[index] for index in accepted_indices],
                episodes[candidate.episode_index],
                key_episode_count=config.key_episode_count,
            )
            accepted = decision.related and decision.confidence >= config.assignment_threshold
            cluster_checks.append(
                {
                    "seed_episode_id": episode_id(episodes[seed_index]),
                    "source_episode_id": episode_id(episodes[last_accepted_index]),
                    "candidate_episode_id": episode_id(episodes[candidate.episode_index]),
                    "bm25_score": candidate.bm25_score,
                    "combined_score": candidate.combined_score,
                    "candidate_reasons": candidate.reasons,
                    "accepted": accepted,
                    "threshold": config.assignment_threshold,
                    "assignment": {
                        "related": decision.related,
                        "relation": decision.relation,
                        "trajectory_hint": decision.trajectory_hint,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "source": decision.source,
                    },
                }
            )
            if accepted:
                accepted_indices.append(candidate.episode_index)
                unassigned.remove(candidate.episode_index)
                last_accepted_index = candidate.episode_index

        if config.include_singletons or len(accepted_indices) > 1:
            trajectories.append(
                build_cluster_payload(
                    episodes,
                    sorted(accepted_indices),
                    cluster_checks,
                    sample_id=sample_id,
                    trajectory_index=len(trajectories) + 1,
                )
            )
    return trajectories


def locomo_trajectories_from_episodes(
    conversation_id: str,
    episodes: list[dict[str, Any]],
    config: ConstructionConfig,
) -> list[TrajectoryMemory]:
    if not episodes:
        return []
    builder_name = config.resolved_trajectory_builder()
    classifier: TitleReasonClusterClassifier
    cluster_config = TitleReasonClusterConfig()
    if builder_name == "locomo_llm":
        try:
            classifier = OpenAICompatibleClusterClassifier(config)
            payloads = build_title_reason_clusters(
                episodes, classifier, sample_id=conversation_id, config=cluster_config
            )
        except (LlmHttpError, OSError, json.JSONDecodeError, ValueError):
            payloads = build_title_reason_clusters(
                episodes,
                LexicalClusterClassifier(),
                sample_id=conversation_id,
                config=TitleReasonClusterConfig(assignment_threshold=0.55),
            )
            builder_name = "locomo"
    else:
        payloads = build_title_reason_clusters(
            episodes,
            LexicalClusterClassifier(),
            sample_id=conversation_id,
            config=TitleReasonClusterConfig(assignment_threshold=0.55),
        )
    results: list[TrajectoryMemory] = []
    for payload in payloads:
        results.append(
            TrajectoryMemory(
                trajectory_id=str(payload["trajectory_id"]),
                conversation_id=conversation_id,
                episode_ids=[str(item) for item in payload.get("episode_ids") or []],
                title=str(payload.get("title") or "trajectory"),
                summary=str(payload.get("summary") or ""),
                relations=list(payload.get("relations") or []),
                metadata={
                    "builder": builder_name,
                    "topic": payload.get("topic"),
                    "reason": payload.get("reason"),
                    "seed_episode_id": payload.get("seed_episode_id"),
                    "construction": payload.get("construction"),
                    "episode_count": payload.get("episode_count"),
                },
            )
        )
    return results


def build_cluster_payload(
    episodes: list[dict[str, Any]],
    accepted_indices: list[int],
    checks: list[dict[str, Any]],
    *,
    sample_id: str,
    trajectory_index: int,
) -> dict[str, Any]:
    accepted_ids = [episode_id(episodes[index]) for index in accepted_indices]
    trajectory_id = f"traj_{sample_id}_{trajectory_index:04d}"
    accepted_checks = [check for check in checks if check.get("accepted")]
    relations = []
    for check in accepted_checks:
        assignment = check.get("assignment") or {}
        relations.append(
            {
                "source_episode_id": check.get("source_episode_id"),
                "target_episode_id": check.get("candidate_episode_id"),
                "relation": assignment.get("relation") or "continuation",
                "confidence": assignment.get("confidence"),
                "reason": assignment.get("reason") or "",
                "bm25_score": check.get("bm25_score"),
            }
        )
    title = compact_text(str(episodes[accepted_indices[0]].get("title") or accepted_ids[0]))[:80]
    summaries = [compact_text(str(episodes[index].get("summary") or "")) for index in accepted_indices]
    summary = compact_text(" ".join(item for item in summaries if item))[:360]
    return {
        "trajectory_id": trajectory_id,
        "sample_id": sample_id,
        "trajectory_index": trajectory_index,
        "title": title or trajectory_id,
        "reason": "title-reason BM25 relation clustering",
        "topic": title,
        "summary": summary or title,
        "seed_episode_id": accepted_ids[0],
        "episode_count": len(accepted_ids),
        "episode_ids": accepted_ids,
        "relations": relations,
        "construction": "title_reason_bm25_single_pass_relation_cluster",
    }


def episode_id(episode: dict[str, Any]) -> str:
    return str(episode.get("episode_id") or episode.get("id") or "")


def episode_title_reason_text(episode: dict[str, Any]) -> str:
    metadata = episode.get("metadata") or {}
    reason = str(metadata.get("reason") or episode.get("reason") or "")
    if reason.startswith("heuristic") or reason.startswith("filled uncovered"):
        reason = ""
    parts = [
        str(episode.get("title") or ""),
        str(metadata.get("topic") or episode.get("topic") or ""),
        str(episode.get("summary") or ""),
        reason,
    ]
    inline = episode.get("memories") or []
    if isinstance(inline, list):
        parts.extend(memory_text(item) if isinstance(item, dict) else str(item) for item in inline[:4])
    return compact_text(" ".join(parts))


def cluster_query_text(episodes: list[dict[str, Any]], *, key_episode_count: int) -> str:
    selected = episodes[:1] + episodes[max(1, len(episodes) - key_episode_count + 1) :]
    seen: set[str] = set()
    lines: list[str] = []
    for episode in selected:
        key = episode_id(episode)
        if key in seen:
            continue
        seen.add(key)
        lines.append(episode_title_reason_text(episode))
    return "\n".join(lines)
