from __future__ import annotations

import unittest

from cogmem_core.recall.progressive import (
    SEARCH_HINT,
    compact_hits,
    empty_progressive_recall,
    format_progressive_prepend,
    is_sufficient,
    next_tools,
    progressive_recall_fields,
    question_layer,
)


class ProgressiveRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = {
            "conversation_id": "sess_001",
            "query": "深圳天气",
            "cognitive_context": {
                "supporting_evidence": [
                    {
                        "item_id": "mem_weather",
                        "text": "今天深圳天气怎么样？",
                        "source": "supporting_evidence",
                        "score": 2.1,
                        "metadata": {"episode_id": "sess_001_ep_0001"},
                    }
                ],
                "current_situation": [
                    {
                        "item_id": "sess_001_ep_0001",
                        "text": "User asks Shenzhen weather.",
                        "source": "current_situation",
                        "score": 1.4,
                        "metadata": {},
                    }
                ],
                "long_term_background": [
                    {
                        "item_id": "traj_weather",
                        "text": "Weather checks in Shenzhen.",
                        "source": "long_term_background",
                        "score": 0.8,
                        "metadata": {"episode_ids": ["sess_001_ep_0001"]},
                    }
                ],
            },
        }

    def test_compact_hits_map_slots_to_kinds_and_follow_up(self) -> None:
        hits = compact_hits(self.context)
        kinds = {hit["kind"] for hit in hits}
        self.assertEqual(kinds, {"memory", "episode", "trajectory"})
        memory = next(hit for hit in hits if hit["kind"] == "memory")
        self.assertEqual(memory["episode_id"], "sess_001_ep_0001")
        self.assertIn("cogmem_get", memory["follow_up"])
        self.assertIn("cogmem_timeline", memory["follow_up"])
        self.assertLessEqual(len(memory["snippet"]), 180)

    def test_question_layer_routes_preference_timeline_and_background(self) -> None:
        self.assertEqual(question_layer("现在怎么叫我"), "preference")
        self.assertEqual(question_layer("这件事的经过是怎样的？"), "timeline")
        self.assertEqual(question_layer("这个课题的长期背景是什么？"), "trajectory")
        self.assertEqual(question_layer("原话具体是哪一句？"), "memory")
        self.assertEqual(question_layer("深圳天气"), "memory")

    def test_preference_query_puts_preference_excerpts_first(self) -> None:
        context = {
            "conversation_id": "sess_001",
            "query": "现在怎么叫我",
            "cognitive_context": {
                "supporting_evidence": [
                    {
                        "item_id": "mem_weather",
                        "text": "今天深圳天气怎么样？",
                        "source": "supporting_evidence",
                        "score": 2.1,
                        "metadata": {},
                    }
                ],
                "preference_memory": [
                    {
                        "item_id": "pref_name",
                        "text": "用户是「王博」。",
                        "source": "preference_memory",
                        "score": 0.9,
                        "metadata": {"status": "active"},
                    }
                ],
                "long_term_background": [
                    {
                        "item_id": "traj_weather",
                        "text": "Weather checks in Shenzhen.",
                        "source": "long_term_background",
                        "score": 0.8,
                        "metadata": {},
                    }
                ],
            },
        }
        fields = progressive_recall_fields(context, query="现在怎么叫我")
        self.assertEqual(fields["question_layer"], "preference")
        self.assertEqual(fields["hits"][0]["kind"], "preference")
        self.assertEqual(fields["next_tools"][0], "cogmem_get")
        self.assertIn("Question layer: preference", fields["openclaw"]["prependContext"])
        self.assertNotIn("cogmem_timeline", fields["next_tools"])

    def test_timeline_query_prefers_episode_and_suggests_timeline(self) -> None:
        fields = progressive_recall_fields(self.context, query="这件事的经历顺序是什么")
        self.assertEqual(fields["question_layer"], "timeline")
        self.assertEqual(fields["hits"][0]["kind"], "episode")
        self.assertEqual(fields["next_tools"][0], "cogmem_timeline")
        self.assertFalse(fields["sufficient"])
        prepend = fields["openclaw"]["prependContext"]
        self.assertIn("cogmem_timeline", prepend)
        self.assertIn("Question layer: timeline", prepend)

    def test_trajectory_query_puts_background_first(self) -> None:
        fields = progressive_recall_fields(self.context, query="这个课题的长期背景是什么")
        self.assertEqual(fields["question_layer"], "trajectory")
        self.assertEqual(fields["hits"][0]["kind"], "trajectory")
        self.assertEqual(fields["next_tools"][0], "cogmem_get")
        self.assertTrue(fields["sufficient"])

    def test_sufficient_false_when_query_is_long(self) -> None:
        hits = compact_hits(self.context)
        self.assertTrue(is_sufficient(hits, "深圳天气"))
        long_query = "我想起来上次我们好像聊过一些关于天气的事情你还记得具体是哪一天哪个城市吗" * 3
        self.assertFalse(is_sufficient(hits, long_query))
        self.assertIn("cogmem_search", next_tools(hits, sufficient=False))

    def test_empty_hits_suggest_search_only(self) -> None:
        self.assertEqual(next_tools([], sufficient=False), ["cogmem_search"])
        empty = empty_progressive_recall("sess_001", "")
        self.assertFalse(empty["sufficient"])
        self.assertEqual(empty["openclaw"]["prependContext"], SEARCH_HINT)

    def test_progressive_fields_do_not_inject_full_context_dump(self) -> None:
        fields = progressive_recall_fields(self.context, query="深圳天气")
        prepend = fields["openclaw"]["prependContext"]
        self.assertIn("CogMem recalled excerpts", prepend)
        self.assertIn("深圳天气", prepend)
        self.assertIn("cogmem_get", prepend)
        self.assertIn("Question layer: memory", prepend)
        self.assertNotIn("Cognitive Context Construction", prepend)
        self.assertTrue(fields["sufficient"])
        self.assertEqual(fields["question_layer"], "memory")
        self.assertEqual(fields["next_tools"][0], "cogmem_get")
        self.assertIn("cogmem_get", fields["next_tools"])
        self.assertNotIn("cogmem_timeline", prepend)

    def test_format_prepend_empty_uses_search_hint(self) -> None:
        text = format_progressive_prepend([], sufficient=False, suggested=["cogmem_search"])
        self.assertEqual(text, SEARCH_HINT)


if __name__ == "__main__":
    unittest.main()
