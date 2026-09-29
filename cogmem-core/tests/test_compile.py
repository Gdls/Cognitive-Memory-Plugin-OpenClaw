from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
import unittest

from cogmem_core import CogMem
from cogmem_core.contracts.recall import ContextMemoryItem
from cogmem_core.recall.bidirectional import HybridRecall, SearchHit
from cogmem_core.recall.compile import compile_cognitive_context
from cogmem_core.construction.trajectories import LexicalClusterClassifier


class CognitiveContextCompileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store_path = Path(__file__).resolve().parent / ".tmp" / f"compile_{uuid4().hex}"
        self.store_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

    def test_identity_query_drops_unrelated_episodes_and_keeps_preferences(self) -> None:
        memory = CogMem(store_path=self.store_path)
        conversation_id = f"sess_{uuid4().hex}"
        memory.append_events(
            conversation_id,
            [
                {
                    "event_type": "user_message",
                    "text": "你叫小弟，我是你的姐夫。不用总是反问。",
                },
                {"event_type": "assistant_message", "text": "好，记住了。我叫小弟。"},
                {
                    "event_type": "user_message",
                    "text": "最近有个紫薯精的抖音视频，很好笑。",
                },
                {"event_type": "assistant_message", "text": "我去搜搜紫薯精。"},
                {"event_type": "user_message", "text": "/setprivacy"},
                {"event_type": "assistant_message", "text": "你是想设置隐私吗？"},
            ],
        )
        memory.build_memory(conversation_id)
        context = memory.retrieve_context(conversation_id, "我应该怎么叫你？")
        packed = context["cognitive_context"]
        prompt = context["prompt_text"]

        pref_text = " ".join(item["text"] for item in packed["preference_memory"])
        self.assertIn("小弟", pref_text)
        self.assertTrue(packed["diagnostics"].get("compiled"))
        self.assertNotIn("紫薯精", prompt)
        self.assertNotIn("/setprivacy", prompt)
        situation = " ".join(item["text"] for item in packed["current_situation"])
        related = " ".join(item["text"] for item in packed["related_experience"])
        self.assertTrue("小弟" in situation or "姐夫" in situation or "小弟" in pref_text)
        self.assertNotIn("紫薯精", situation + related)
        self.assertLess(packed["diagnostics"].get("prompt_item_chars") or 0, 1600)

    def test_weather_query_still_keeps_shenzhen_evidence(self) -> None:
        memory = CogMem(store_path=self.store_path)
        conversation_id = f"sess_{uuid4().hex}"
        memory.append_events(
            conversation_id,
            [
                {"event_type": "user_message", "text": "今天深圳天气怎么样？"},
                {"event_type": "assistant_message", "text": "我去查一下深圳的天气。"},
                {"event_type": "user_message", "text": "这台服务器有几块 GPU？"},
                {"event_type": "assistant_message", "text": "一共八张 A100。"},
            ],
        )
        memory.build_memory(conversation_id)
        prompt = memory.format_prompt_context(conversation_id, "用户之前问过什么天气？")
        self.assertIn("深圳天气", prompt)


    def test_update_relation_keeps_later_episode_in_situation(self) -> None:
        context = compile_cognitive_context(
            query="现在还帮我做工作项目吗？",
            conversation_id="sess_update",
            pack=_pack_with_update(),
            preference_hits=[],
            temporal_hits=[],
            execution_hits=[],
            trajectories=[
                {
                    "trajectory_id": "traj_role",
                    "episode_ids": ["ep_old", "ep_new"],
                    "relations": [
                        {
                            "source_episode_id": "ep_old",
                            "target_episode_id": "ep_new",
                            "relation": "update",
                        }
                    ],
                }
            ],
        )
        situation_ids = [item.item_id for item in context.current_situation]
        related_text = " ".join(item.text for item in context.related_experience)
        self.assertIn("ep_new", situation_ids)
        self.assertNotIn("ep_old", situation_ids)
        self.assertIn("ep_old", [item.item_id for item in context.related_experience])
        self.assertIn("已被后续更新", related_text)
        self.assertGreaterEqual(context.diagnostics.get("arbitrated_updates") or 0, 1)

    def test_conflict_keeps_both_with_note(self) -> None:
        pack = HybridRecall(
            trajectories=[],
            episodes=[
                SearchHit(
                    "ep_a",
                    0.9,
                    {"title": "计划去深圳", "summary": "下周去深圳出差。", "_retrieval_role": "conflict"},
                ),
                SearchHit(
                    "ep_b",
                    0.8,
                    {"title": "改去广州", "summary": "下周改去广州。", "_retrieval_role": "conflict"},
                ),
            ],
            memories=[],
            relation_notes=[
                {
                    "source_episode_id": "ep_a",
                    "target_episode_id": "ep_b",
                    "relation": "conflict",
                }
            ],
            diagnostics={},
        )
        context = compile_cognitive_context(
            query="下周去哪出差？",
            conversation_id="sess_conflict",
            pack=pack,
            preference_hits=[],
            temporal_hits=[],
            execution_hits=[],
            trajectories=[
                {
                    "trajectory_id": "traj_trip",
                    "relations": [
                        {
                            "source_episode_id": "ep_a",
                            "target_episode_id": "ep_b",
                            "relation": "conflict",
                        }
                    ],
                }
            ],
        )
        texts = " ".join(
            item.text for item in context.current_situation + context.related_experience
        )
        self.assertIn("深圳", texts)
        self.assertIn("广州", texts)
        self.assertIn("存在冲突", texts)

    def test_compiled_text_strips_tool_call_and_duplicate_title(self) -> None:
        title = "你是我新建的智能体机器人，你叫小弟"
        pack = HybridRecall(
            trajectories=[],
            episodes=[
                SearchHit(
                    "ep_name",
                    0.95,
                    {
                        "title": title,
                        "summary": f"{title} tool_call write path=/tmp/openclaw/foo.json success: true",
                        "_retrieval_role": "primary",
                    },
                )
            ],
            memories=[],
            relation_notes=[],
            diagnostics={},
        )
        context = compile_cognitive_context(
            query="你叫小弟吗？",
            conversation_id="sess_noise",
            pack=pack,
            preference_hits=[],
            temporal_hits=[],
            execution_hits=[],
        )
        text = " ".join(item.text for item in context.current_situation)
        self.assertIn("小弟", text)
        self.assertNotIn("tool_call", text)
        self.assertNotIn("path=/", text)
        self.assertEqual(text.count("你叫小弟"), 1)

    def test_later_role_preference_wins_over_earlier_same_topic(self) -> None:
        pack = HybridRecall(
            trajectories=[],
            episodes=[],
            memories=[],
            relation_notes=[],
            diagnostics={},
        )
        context = compile_cognitive_context(
            query="现在人设是什么？",
            conversation_id="sess_role",
            pack=pack,
            preference_hits=[
                ContextMemoryItem(
                    item_id="pref_old",
                    text="用户希望助手处理工作项目。",
                    source="preference_memory",
                    score=0.84,
                    metadata={
                        "preference_type": "role",
                        "evidence_ids": ["m1"],
                        "evidence_episode_ids": ["ep_old"],
                    },
                ),
                ContextMemoryItem(
                    item_id="pref_new",
                    text="用户希望助手不处理工作项目。",
                    source="preference_memory",
                    score=0.84,
                    metadata={
                        "preference_type": "role",
                        "evidence_ids": ["m2"],
                        "evidence_episode_ids": ["ep_new"],
                    },
                ),
                ContextMemoryItem(
                    item_id="pref_life",
                    text="用户希望助手做生活情绪调节。",
                    source="preference_memory",
                    score=0.84,
                    metadata={
                        "preference_type": "role",
                        "evidence_ids": ["m3"],
                        "evidence_episode_ids": ["ep_new"],
                    },
                ),
            ],
            temporal_hits=[],
            execution_hits=[],
            episodes=[
                {"episode_id": "ep_old", "start_time": "2024-01-01T00:00:00"},
                {"episode_id": "ep_new", "start_time": "2024-06-01T00:00:00"},
            ],
        )
        pref_text = " ".join(item.text for item in context.preference_memory)
        self.assertIn("不处理工作", pref_text)
        self.assertNotIn("助手处理工作项目", pref_text)
        self.assertIn("生活情绪", pref_text)
        self.assertGreaterEqual(context.diagnostics.get("preference_superseded") or 0, 1)

    def test_lexical_classifier_marks_later_negation_as_update(self) -> None:
        decision = LexicalClusterClassifier().classify_title_reason_profile(
            [{"title": "处理工作项目", "summary": "用户希望助手处理工作上的项目"}],
            {"title": "不碰工作", "summary": "用户不用去处理那些工作上的项目"},
            key_episode_count=5,
        )
        self.assertTrue(decision.related)
        self.assertEqual(decision.relation, "update")


def _pack_with_update() -> HybridRecall:
    return HybridRecall(
        trajectories=[
            SearchHit(
                "traj_role",
                0.8,
                {
                    "title": "助手角色",
                    "summary": "从接项目改成不碰工作",
                    "episode_ids": ["ep_old", "ep_new"],
                    "relations": [
                        {
                            "source_episode_id": "ep_old",
                            "target_episode_id": "ep_new",
                            "relation": "update",
                        }
                    ],
                },
            )
        ],
        episodes=[
            SearchHit(
                "ep_old",
                0.9,
                {
                    "title": "接工作项目",
                    "summary": "用户让助手处理工作上的项目。",
                    "start_time": "2024-01-01T00:00:00",
                    "_retrieval_role": "primary",
                },
            ),
            SearchHit(
                "ep_new",
                0.85,
                {
                    "title": "不碰工作",
                    "summary": "用户说不用去处理那些工作上的项目。",
                    "start_time": "2024-06-01T00:00:00",
                    "_retrieval_role": "latest_update",
                },
            ),
        ],
        memories=[],
        relation_notes=[
            {
                "trajectory_id": "traj_role",
                "source_episode_id": "ep_old",
                "target_episode_id": "ep_new",
                "relation": "update",
            }
        ],
        diagnostics={},
    )


if __name__ == "__main__":
    unittest.main()
