from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
import unittest

from cogmem_core import CogMem, ConstructionConfig
from cogmem_core.construction.episodes import (
    EpisodeSegment,
    HeuristicEpisodeSegmenter,
    build_rolling_episodes,
    parse_line_segments,
)
from cogmem_core.construction.trajectories import LexicalClusterClassifier, build_title_reason_clusters


class LocomoBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store_path = Path(__file__).resolve().parent / ".tmp" / f"locomo_{uuid4().hex}"
        self.store_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

    def test_line_segment_parser_covers_mem_ranges(self) -> None:
        text = (
            "mem0-mem2 | Weather check | Shenzhen weather | User asks Shenzhen weather. | same event | 0.86\n"
            "mem3-mem5 | GPU inventory | GPU hardware | User asks GPU count. | topic shift | 0.81\n"
        )
        segments = parse_line_segments(text)
        self.assertEqual([(item.start, item.end) for item in segments], [(0, 2), (3, 5)])
        self.assertEqual(segments[0].topic, "Shenzhen weather")

    def test_rolling_window_commits_head_and_keeps_tail(self) -> None:
        class FixedSegmenter:
            def segment(self, memories):
                last = len(memories) - 1
                mid = max(0, last // 2)
                return [
                    EpisodeSegment(0, mid, "head", "first half", 0.9, "head", "head"),
                    EpisodeSegment(mid + 1, last, "tail", "second half", 0.9, "tail", "tail"),
                ]

        memories = [{"memory_id": f"m{i}", "text": f"turn {i} about adapters"} for i in range(10)]
        episodes = build_rolling_episodes(
            memories, FixedSegmenter(), sample_id="sess", config=None
        )
        self.assertGreaterEqual(len(episodes), 1)
        covered = sum(len(item["memory_ids"]) for item in episodes)
        self.assertEqual(covered, 10)

    def test_heuristic_splits_unrelated_user_topics(self) -> None:
        memories = [
            {"memory_id": "a", "text": "今天深圳天气怎么样？", "metadata": {"event_type": "user_message"}},
            {"memory_id": "b", "text": "深圳今天多云，大约二十六度。", "metadata": {"event_type": "assistant_message"}},
            {"memory_id": "c", "text": "这台服务器有几块 GPU，显存多少？", "metadata": {"event_type": "user_message"}},
            {"memory_id": "d", "text": "机器上有八张 A100，每张八十 GB。", "metadata": {"event_type": "assistant_message"}},
        ]
        segments = HeuristicEpisodeSegmenter().segment(memories)
        self.assertGreaterEqual(len(segments), 2)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[-1].end, 3)

    def test_build_memory_uses_locomo_metadata(self) -> None:
        memory = CogMem(store_path=self.store_path, construction_config=ConstructionConfig())
        conversation_id = "sess_locomo"
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "今天深圳天气怎么样？"},
                {"event_type": "assistant_message", "text": "深圳今天多云。"},
                {"event_type": "user_message", "text": "这台服务器有几块 GPU？"},
                {"event_type": "assistant_message", "text": "一共八张 A100。"},
            ],
        )
        build = memory.build_memory(conversation_id)
        self.assertGreaterEqual(build["episodes"], 2)
        episodes = memory.store.load_episodes(conversation_id)
        self.assertEqual(episodes[0]["metadata"]["builder"], "locomo")
        trajectories = memory.store.load_trajectories(conversation_id)
        self.assertGreaterEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0]["metadata"]["builder"], "locomo")
        self.assertEqual(
            trajectories[0]["metadata"]["construction"],
            "title_reason_bm25_single_pass_relation_cluster",
        )

    def test_placeholder_backend_still_emits_one_runtime_trajectory(self) -> None:
        config = ConstructionConfig(episode_builder="placeholder", trajectory_builder="placeholder")
        memory = CogMem(store_path=self.store_path, construction_config=config)
        conversation_id = "sess_placeholder"
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "hello"},
                {"event_type": "assistant_message", "text": "hi"},
            ],
        )
        build = memory.build_memory(conversation_id)
        self.assertEqual(build["episodes"], 1)
        self.assertEqual(build["trajectories"], 1)
        trajectory = memory.store.load_trajectories(conversation_id)[0]
        self.assertIn("runtime trajectory", trajectory["title"])

    def test_lexical_cluster_joins_related_and_splits_unrelated(self) -> None:
        episodes = [
            {"episode_id": "e1", "title": "Shenzhen weather", "summary": "ask Shenzhen weather today"},
            {"episode_id": "e2", "title": "Shenzhen forecast", "summary": "cloudy Shenzhen weather update"},
            {"episode_id": "e3", "title": "GPU inventory", "summary": "count A100 GPU cards and VRAM"},
        ]
        trajectories = build_title_reason_clusters(
            episodes, LexicalClusterClassifier(), sample_id="s"
        )
        grouped = {tuple(item["episode_ids"]) for item in trajectories}
        self.assertTrue(any("e1" in ids and "e2" in ids for ids in grouped))
        self.assertTrue(any(ids == ("e3",) or "e3" in ids and "e1" not in ids for ids in grouped))


if __name__ == "__main__":
    unittest.main()
