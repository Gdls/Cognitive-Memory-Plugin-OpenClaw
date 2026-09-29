from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
import unittest

from cogmem_core import CogMem
from cogmem_core.recall.progressive import compact_hits, format_progressive_prepend
from cogmem_core.status import apply_cognitive_status, apply_preference_status, item_status, query_intent


class CognitiveStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store_path = Path(__file__).resolve().parent / ".tmp" / f"status_{uuid4().hex}"
        self.store_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

    def test_update_relation_marks_source_episode_superseded(self) -> None:
        episodes = [
            {"episode_id": "ep_old", "summary": "叫我小王", "memory_ids": ["m1"], "metadata": {}},
            {"episode_id": "ep_new", "summary": "纠正一下，叫我王博", "memory_ids": ["m2"], "metadata": {}},
        ]
        memories = [
            {"memory_id": "m1", "text": "叫我小王", "metadata": {"event_type": "user_message"}},
            {"memory_id": "m2", "text": "纠正一下，叫我王博", "metadata": {"event_type": "user_message"}},
        ]
        trajectories = [
            {
                "trajectory_id": "traj_name",
                "episode_ids": ["ep_old", "ep_new"],
                "relations": [
                    {
                        "source_episode_id": "ep_old",
                        "target_episode_id": "ep_new",
                        "relation": "correction",
                    }
                ],
            }
        ]
        report = apply_cognitive_status(episodes, trajectories, memories)
        self.assertEqual(report["superseded_by"]["ep_old"], "ep_new")
        self.assertEqual(item_status(episodes[0]), "superseded")
        self.assertEqual(item_status(episodes[1]), "active")
        self.assertEqual(item_status(memories[0]), "superseded")
        self.assertEqual(item_status(memories[1]), "active")
        self.assertEqual(memories[0]["metadata"].get("event_type"), "user_message")

    def test_later_address_preference_supersedes_earlier(self) -> None:
        preferences = [
            {
                "memory_id": "pref_old",
                "preference_type": "address",
                "content": "用户是「小王」。",
                "confidence": 0.86,
                "evidence_ids": ["m1"],
                "metadata": {"evidence_episode_ids": ["ep_old"]},
            },
            {
                "memory_id": "pref_new",
                "preference_type": "address",
                "content": "用户是「王博」。",
                "confidence": 0.86,
                "evidence_ids": ["m2"],
                "metadata": {"evidence_episode_ids": ["ep_new"]},
            },
        ]
        apply_preference_status(
            preferences,
            [
                {"episode_id": "ep_old", "end_time": "2026-09-01T10:00:00+00:00"},
                {"episode_id": "ep_new", "end_time": "2026-09-01T11:00:00+00:00"},
            ],
        )
        by_id = {item["memory_id"]: item for item in preferences}
        self.assertEqual(by_id["pref_old"]["status"], "superseded")
        self.assertEqual(by_id["pref_new"]["status"], "active")

    def test_address_correction_demo_current_name_is_wangbo(self) -> None:
        memory = CogMem(store_path=self.store_path)
        conversation_id = f"sess_{uuid4().hex}"
        memory.append_events(
            conversation_id,
            [
                {
                    "event_type": "user_message",
                    "text": "叫我小王",
                    "timestamp": "2026-09-01T10:00:00+00:00",
                },
                {
                    "event_type": "assistant_message",
                    "text": "好的，小王。",
                    "timestamp": "2026-09-01T10:00:01+00:00",
                },
                {
                    "event_type": "user_message",
                    "text": "纠正一下，叫我王博",
                    "timestamp": "2026-09-01T11:00:00+00:00",
                },
                {
                    "event_type": "assistant_message",
                    "text": "好的，记住了。",
                    "timestamp": "2026-09-01T11:00:01+00:00",
                },
            ],
        )
        memory.build_memory(conversation_id)

        preferences = memory.store.load_preferences(conversation_id)
        address = [item for item in preferences if item.get("preference_type") == "address"]
        self.assertGreaterEqual(len(address), 2)
        active = [item for item in address if item.get("status") == "active"]
        superseded = [item for item in address if item.get("status") == "superseded"]
        self.assertTrue(any("王博" in str(item.get("content") or "") for item in active))
        self.assertTrue(any("小王" in str(item.get("content") or "") for item in superseded))

        current = memory.retrieve_context(conversation_id, "现在怎么叫我")
        packed = current["cognitive_context"]
        pref_text = " ".join(item["text"] for item in packed["preference_memory"])
        situation = " ".join(item["text"] for item in packed["current_situation"])
        self.assertIn("王博", pref_text)
        self.assertNotIn("小王", pref_text)
        self.assertIn("王博", current["prompt_text"])
        self.assertNotIn("小王", situation)
        self.assertEqual(packed["diagnostics"].get("query_intent"), "current_state")

        history = memory.retrieve_context(conversation_id, "以前叫我什么")
        history_pref = " ".join(
            item["text"] for item in history["cognitive_context"]["preference_memory"]
        )
        self.assertIn("小王", history_pref)
        self.assertEqual(history["cognitive_context"]["diagnostics"].get("query_intent"), "history_process")

        excerpts = compact_hits(current, query="现在怎么叫我")
        self.assertFalse(any(hit.get("status") == "superseded" for hit in excerpts))
        history_excerpts = compact_hits(history, query="以前叫我什么")
        self.assertTrue(
            any("小王" in str(hit.get("snippet") or "") for hit in history_excerpts)
            or "小王" in history_pref
        )

    def test_intent_identity_history_beats_current_name_query(self) -> None:
        self.assertEqual(query_intent("现在怎么叫我"), "current_state")
        self.assertEqual(query_intent("以前叫我什么"), "history_process")
        self.assertEqual(query_intent("用户之前问过什么天气？"), "evidence_lookup")

    def test_progressive_excerpts_tag_superseded_on_history_query(self) -> None:
        context = {
            "conversation_id": "sess_001",
            "query": "以前叫我什么",
            "cognitive_context": {
                "preference_memory": [
                    {
                        "item_id": "pref_new",
                        "text": "用户是「王博」。",
                        "source": "preference_memory",
                        "score": 1.0,
                        "metadata": {"status": "active", "preference_type": "address"},
                    },
                    {
                        "item_id": "pref_old",
                        "text": "（已被后续更新）用户是「小王」。",
                        "source": "preference_memory",
                        "score": 0.4,
                        "metadata": {"status": "superseded", "preference_type": "address"},
                    },
                ]
            },
        }
        hits = compact_hits(context, query="以前叫我什么")
        statuses = {hit["id"]: hit["status"] for hit in hits}
        self.assertEqual(statuses["pref_old"], "superseded")
        prepend = format_progressive_prepend(hits, sufficient=True, suggested=["cogmem_get"])
        self.assertIn("status=superseded", prepend)


if __name__ == "__main__":
    unittest.main()
