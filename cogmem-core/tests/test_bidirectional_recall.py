from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
import unittest

from cogmem_core import CogMem
from cogmem_core.recall.bidirectional import hybrid_recall


class BidirectionalRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store_path = Path(__file__).resolve().parent / ".tmp" / f"recall_{uuid4().hex}"
        self.store_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

    def test_hybrid_merges_top_down_path_with_bottom_up_fact(self) -> None:
        memories = [
            {"memory_id": "mem_weather", "text": "今天深圳天气怎么样？", "timestamp": "t1"},
            {"memory_id": "mem_gpu", "text": "这台服务器安装了八张 A100 GPU。", "timestamp": "t2"},
        ]
        episodes = [
            {
                "episode_id": "ep_weather",
                "title": "Ask Shenzhen weather",
                "summary": "User asks about Shenzhen weather today.",
                "memory_ids": ["mem_weather"],
            },
            {
                "episode_id": "ep_gpu",
                "title": "GPU inventory",
                "summary": "User asks how many GPU cards the server has.",
                "memory_ids": ["mem_gpu"],
            },
        ]
        trajectories = [
            {
                "trajectory_id": "traj_weather",
                "title": "Weather checks",
                "summary": "Repeated local weather questions.",
                "episode_ids": ["ep_weather"],
                "relations": [],
            },
            {
                "trajectory_id": "traj_gpu",
                "title": "Cluster hardware",
                "summary": "Server GPU and VRAM inventory.",
                "episode_ids": ["ep_gpu"],
                "relations": [],
            },
        ]

        pack = hybrid_recall(
            "用户之前问过什么天气？",
            memories=memories,
            episodes=episodes,
            trajectories=trajectories,
        )

        self.assertEqual(pack.diagnostics["retrieval_mode"], "hybrid")
        self.assertGreaterEqual(pack.diagnostics["merged_trajectory_count"], 1)
        self.assertEqual(pack.trajectories[0].item_id, "traj_weather")
        self.assertEqual(pack.episodes[0].item_id, "ep_weather")
        self.assertEqual(pack.memories[0].snippet_id, "mem_weather")
        self.assertIn("top_down", pack.trajectories[0].payload.get("_retrieval_sources") or [])
        self.assertIn("bottom_up", pack.trajectories[0].payload.get("_retrieval_sources") or [])

    def test_content_terms_beat_generic_chinese_question_words(self) -> None:
        memories = [
            {"memory_id": "mem_weather", "text": "今天深圳天气怎么样？"},
            {
                "memory_id": "mem_crash",
                "text": "请问之前发生了什么？是会话被中断了吗？我现在恢复正常了。",
            },
            {
                "memory_id": "mem_gpu",
                "text": "好的，以下是您当前提问的上下文：提问内容：我这台服务器安装了几块 GPU 卡，什么型号，显存多少？",
            },
        ]
        episodes = [
            {
                "episode_id": "ep_weather",
                "title": "Ask Shenzhen weather",
                "summary": "User asks about Shenzhen weather today.",
                "memory_ids": ["mem_weather"],
            },
            {
                "episode_id": "ep_crash",
                "title": "Session recovered",
                "summary": "Assistant apologizes after a crash.",
                "memory_ids": ["mem_crash"],
            },
            {
                "episode_id": "ep_gpu",
                "title": "Show current question context",
                "summary": "Dump the current GPU question as context.",
                "memory_ids": ["mem_gpu"],
            },
        ]
        trajectories = [
            {
                "trajectory_id": "traj_weather",
                "title": "Weather checks",
                "summary": "Repeated local weather questions.",
                "episode_ids": ["ep_weather"],
                "relations": [],
            },
            {
                "trajectory_id": "traj_crash",
                "title": "Crash recovery",
                "summary": "Bot recovered after being overwritten.",
                "episode_ids": ["ep_crash"],
                "relations": [],
            },
            {
                "trajectory_id": "traj_gpu",
                "title": "GPU context dump",
                "summary": "Show current question context about GPU cards.",
                "episode_ids": ["ep_gpu"],
                "relations": [],
            },
        ]
        pack = hybrid_recall(
            "用户之前问过什么天气？",
            memories=memories,
            episodes=episodes,
            trajectories=trajectories,
        )
        self.assertEqual(pack.trajectories[0].item_id, "traj_weather")
        self.assertEqual(pack.episodes[0].item_id, "ep_weather")
        self.assertEqual(pack.memories[0].snippet_id, "mem_weather")
        evidence_ids = [item.snippet_id for item in pack.memories]
        self.assertIn("mem_weather", evidence_ids)
        if "mem_gpu" in evidence_ids:
            self.assertLess(evidence_ids.index("mem_weather"), evidence_ids.index("mem_gpu"))
        if "mem_crash" in evidence_ids:
            self.assertLess(evidence_ids.index("mem_weather"), evidence_ids.index("mem_crash"))

    def test_bottom_up_recovers_fact_missing_from_trajectory_summary(self) -> None:
        memories = [
            {
                "memory_id": "mem_detail",
                "text": "The leftover cake was stored in the blue fridge on Maple Street.",
                "timestamp": "t1",
            },
            {
                "memory_id": "mem_other",
                "text": "We booked a conference room for the adapter review.",
                "timestamp": "t2",
            },
        ]
        episodes = [
            {
                "episode_id": "ep_home",
                "title": "Evening at home",
                "summary": "Casual chat after dinner.",
                "memory_ids": ["mem_detail"],
            },
            {
                "episode_id": "ep_work",
                "title": "Adapter review",
                "summary": "Project Alpha adapter meeting.",
                "memory_ids": ["mem_other"],
            },
        ]
        trajectories = [
            {
                "trajectory_id": "traj_home",
                "title": "Home life",
                "summary": "Everyday household conversation.",
                "episode_ids": ["ep_home"],
                "relations": [],
            },
            {
                "trajectory_id": "traj_work",
                "title": "Work projects",
                "summary": "Adapter and trajectory engineering work.",
                "episode_ids": ["ep_work"],
                "relations": [],
            },
        ]

        pack = hybrid_recall(
            "Where was the leftover cake stored?",
            memories=memories,
            episodes=episodes,
            trajectories=trajectories,
        )
        self.assertEqual(pack.memories[0].snippet_id, "mem_detail")
        self.assertEqual(pack.episodes[0].item_id, "ep_home")
        self.assertEqual(pack.trajectories[0].item_id, "traj_home")
        self.assertIn("bottom_up", pack.trajectories[0].payload.get("_retrieval_sources") or [])

    def test_runtime_recall_fills_cognitive_context_slots(self) -> None:
        memory = CogMem(store_path=self.store_path)
        conversation_id = f"sess_{uuid4().hex}"
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "今天深圳天气怎么样？"},
                {"event_type": "assistant_message", "text": "我去查一下深圳的天气。"},
                {"event_type": "user_message", "text": "这台服务器有几块 GPU，显存多少？"},
                {"event_type": "assistant_message", "text": "机器上有八张 A100。"},
            ],
        )
        memory.build_memory(conversation_id)
        context = memory.retrieve_context(conversation_id, "用户之前问过什么天气？")
        packed = context["cognitive_context"]
        diagnostics = packed["diagnostics"]

        self.assertEqual(diagnostics["retrieval_mode"], "hybrid")
        self.assertGreaterEqual(diagnostics["merged_episode_count"], 1)
        self.assertGreaterEqual(diagnostics["merged_trajectory_count"], 1)
        evidence = " ".join(item["text"] for item in packed["supporting_evidence"])
        self.assertIn("深圳天气", evidence)
        gpu_index = evidence.find("GPU")
        weather_index = evidence.find("深圳天气")
        if gpu_index >= 0:
            self.assertLess(weather_index, gpu_index)
        situation = " ".join(item["text"] for item in packed["current_situation"])
        self.assertTrue("天气" in situation or "深圳" in situation or "weather" in situation.lower())
        background_ids = [item["item_id"] for item in packed["long_term_background"]]
        self.assertTrue(background_ids)
        self.assertFalse(any(item_id.startswith("mem_") for item_id in background_ids))


if __name__ == "__main__":
    unittest.main()
