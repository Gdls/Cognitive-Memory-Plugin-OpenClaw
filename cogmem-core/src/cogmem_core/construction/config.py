from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class ConstructionConfig:
    """Offline construction backends.

    ``locomo`` ports the LoCoMo rolling-episode + BM25 cluster loop with
    non-LLM judges. ``locomo_llm`` uses the same loop with an OpenAI-compatible
    model. ``placeholder`` keeps the original chunk / single-trajectory builders.
    """

    episode_builder: str = "locomo"
    trajectory_builder: str = "locomo"
    continuity_mode: str = "task_execution"
    episode_batch_size: int = 30
    episode_tail_keep: int = 2
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> ConstructionConfig:
        return cls(
            episode_builder=os.environ.get("COGMEM_EPISODE_BUILDER", "locomo"),
            trajectory_builder=os.environ.get("COGMEM_TRAJECTORY_BUILDER", "locomo"),
            continuity_mode=os.environ.get("COGMEM_CONTINUITY_MODE", "task_execution"),
            episode_batch_size=int(os.environ.get("COGMEM_EPISODE_BATCH_SIZE", "30")),
            episode_tail_keep=int(os.environ.get("COGMEM_EPISODE_TAIL_KEEP", "2")),
            llm_base_url=os.environ.get("COGMEM_LLM_BASE_URL") or None,
            llm_api_key=os.environ.get("COGMEM_LLM_API_KEY") or None,
            llm_model=os.environ.get("COGMEM_LLM_MODEL") or None,
            llm_timeout_seconds=float(os.environ.get("COGMEM_LLM_TIMEOUT", "120")),
        )

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    def resolved_episode_builder(self) -> str:
        if self.episode_builder == "locomo_llm" and not self.llm_ready:
            return "locomo"
        return self.episode_builder

    def resolved_trajectory_builder(self) -> str:
        if self.trajectory_builder == "locomo_llm" and not self.llm_ready:
            return "locomo"
        return self.trajectory_builder
