from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
import unittest

from cogmem_core import (
    BuildRequest,
    CogMem,
    ConstructionService,
    RecallRequest,
    RecallService,
)
from cogmem_core.repositories import LocalJsonStore


class CogMemCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store_path = Path(__file__).resolve().parent / ".tmp" / unique_conversation_id()
        self.store_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

    def test_append_events_makes_temporal_memory_retrievable(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        result = memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "I prefer concise answers."},
                {"event_type": "assistant_message", "text": "I will keep answers concise."},
            ],
        )

        self.assertEqual(result["appended"], 2)
        context = memory.retrieve_context(conversation_id, "concise answers")
        self.assertGreaterEqual(len(context["cognitive_context"]["temporal_memory"]), 1)
        self.assertIn("Cognitive Context Construction", context["prompt_text"])

    def test_build_memory_moves_temporal_events_into_long_term_store(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        memory.append_events(
            conversation_id,
            [{"event_type": "user_message", "text": "The deployment target is Docker."}],
        )

        build = memory.build_memory(conversation_id)
        self.assertEqual(build["processed_events"], 1)
        status = memory.get_status(conversation_id)
        self.assertEqual(status["temporal_events"], 0)
        self.assertEqual(status["memory_objects"], 1)
        self.assertEqual(status["episodes"], 1)
        self.assertEqual(status["trajectories"], 1)

        context = memory.retrieve_context(conversation_id, "Docker deployment")
        self.assertGreaterEqual(len(context["cognitive_context"]["long_term_background"]), 1)
        self.assertGreaterEqual(len(context["cognitive_context"]["current_situation"]), 1)

        episodes = memory.store.load_episodes(conversation_id)
        episode_id = episodes[0]["episode_id"]
        item = memory.get_item(episode_id, kind="episode")
        self.assertTrue(item["found"])
        timeline = memory.get_timeline(episode_id)
        self.assertTrue(timeline["found"])
        self.assertGreaterEqual(len(timeline["memories"]), 1)
        store_status = memory.get_status()
        self.assertIn(conversation_id, store_status["conversation_ids"])

    def test_force_rebuild_uses_raw_events_after_temporal_is_cleared(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        memory.append_events(
            conversation_id,
            [{"event_type": "user_message", "text": "The deployment target is Docker."}],
        )
        memory.build_memory(conversation_id)
        rebuilt = memory.build_memory(conversation_id, force_rebuild=True)
        self.assertEqual(rebuilt["processed_events"], 1)
        self.assertEqual(rebuilt["total_memories"], 1)
        self.assertGreaterEqual(rebuilt["episodes"], 1)

    def test_build_memory_organizes_multiple_memories_into_episode_trajectory_context(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "Project Alpha needs a LoCoMo adapter."},
                {"event_type": "assistant_message", "text": "The adapter should produce episodes."},
                {"event_type": "user_message", "text": "The retrieval path should use trajectories."},
            ],
        )
        build = memory.build_memory(conversation_id)

        self.assertGreaterEqual(build["episodes"], 1)
        self.assertGreaterEqual(build["trajectories"], 1)

        context = memory.retrieve_context(conversation_id, "LoCoMo trajectories")
        diagnostics = context["cognitive_context"]["diagnostics"]
        self.assertEqual(diagnostics["retrieval_mode"], "hybrid")
        self.assertGreaterEqual(diagnostics["episode_candidates"], 1)
        self.assertGreaterEqual(diagnostics["trajectory_candidates"], 1)
        self.assertTrue(
            any(
                item.get("text")
                for item in context["cognitive_context"]["long_term_background"]
            )
        )

    def test_preference_derivation_compiles_style_guidance(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        memory.append_events(
            conversation_id,
            [{"event_type": "user_message", "text": "我偏好先给结论再给细节。"}],
        )
        memory.build_memory(conversation_id)
        result = memory.derive_preferences(conversation_id)

        self.assertEqual(result["candidate_preferences"], 1)
        context = memory.retrieve_context(conversation_id, "回答偏好")
        self.assertGreaterEqual(len(context["cognitive_context"]["preference_memory"]), 1)

    def test_construction_and_recall_services_are_independently_callable(self) -> None:
        conversation_id = unique_conversation_id()
        repository = LocalJsonStore(self.store_path)
        memory = CogMem(repository=repository)
        memory.append_events(
            conversation_id,
            [{"event_type": "user_message", "text": "The memory layer runs offline."}],
        )

        build = ConstructionService(repository).build(BuildRequest(conversation_id))
        context = RecallService(repository).recall(
            RecallRequest(conversation_id=conversation_id, query="offline memory")
        )

        self.assertEqual(build.processed_events, 1)
        self.assertGreaterEqual(len(context.cognitive_context.long_term_background), 1)
        self.assertIn("Cognitive Context Construction", context.prompt_text)

    def test_chinese_query_matches_partial_sentence(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "今天深圳天气怎么样？"},
                {"event_type": "assistant_message", "text": "我去查一下深圳的天气。"},
            ],
        )
        memory.build_memory(conversation_id)
        prompt = memory.format_prompt_context(conversation_id, "用户之前问过什么天气？")
        self.assertIn("深圳天气", prompt)

    def test_weather_query_ranks_weather_above_unrelated_chinese(self) -> None:
        conversation_id = unique_conversation_id()
        memory = CogMem(store_path=self.store_path)
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "今天深圳天气怎么样？"},
                {
                    "event_type": "assistant_message",
                    "text": "我这台服务器安装了几块 GPU 卡，什么型号，显存多少？",
                },
            ],
        )
        memory.build_memory(conversation_id)
        context = memory.retrieve_context(conversation_id, "用户之前问过什么天气？")
        evidence = " ".join(
            item["text"] for item in context["cognitive_context"]["supporting_evidence"]
        )
        self.assertIn("深圳天气", evidence)
        weather_index = evidence.find("深圳天气")
        gpu_index = evidence.find("GPU")
        if gpu_index >= 0:
            self.assertLess(weather_index, gpu_index)


def unique_conversation_id() -> str:
    return f"test_{uuid4().hex}"


if __name__ == "__main__":
    unittest.main()
