from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4
import unittest

from cogmem_core import CogMem
from cogmem_core.preferences import derive_candidate_preferences


class PreferenceMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store_path = Path(__file__).resolve().parent / ".tmp" / f"pref_{uuid4().hex}"
        self.store_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

    def test_extracts_address_role_and_constraints_from_user_turns(self) -> None:
        candidates = derive_candidate_preferences(
            "sess",
            [
                {
                    "memory_id": "m1",
                    "text": "你是我新建的智能体机器人，你叫小弟，我是你的姐夫",
                    "metadata": {"event_type": "user_message", "speaker": "姐夫"},
                },
                {
                    "memory_id": "m2",
                    "text": "说话时还是要有称呼的，显着咱俩关系亲近",
                    "metadata": {"event_type": "user_message"},
                },
                {
                    "memory_id": "m3",
                    "text": "你是个生活情绪调节小助手，不用去处理那些工作上的项目",
                    "metadata": {"event_type": "user_message"},
                },
                {
                    "memory_id": "m4",
                    "text": "不用总是反问，除非有必要",
                    "metadata": {"event_type": "user_message"},
                },
                {
                    "memory_id": "m5",
                    "text": "好的，以下是您当前提问的上下文：GPU 显存多少？",
                    "metadata": {"event_type": "assistant_message", "speaker": "assistant"},
                },
            ],
        )
        contents = " ".join(item.content for item in candidates)
        types = {item.preference_type for item in candidates}
        self.assertIn("address", types)
        self.assertIn("小弟", contents)
        self.assertIn("姐夫", contents)
        self.assertIn("反问", contents)
        self.assertIn("生活情绪", contents)
        self.assertNotIn("GPU", contents)
        for item in candidates:
            self.assertLessEqual(len(item.content), 120)

    def test_build_writes_preferences_used_as_guidance_not_evidence(self) -> None:
        memory = CogMem(store_path=self.store_path)
        conversation_id = f"sess_{uuid4().hex}"
        memory.append_events(
            conversation_id,
            [
                {
                    "event_type": "user_message",
                    "text": "你叫小弟，我是你的姐夫。我偏好先给结论再给细节。",
                },
                {"event_type": "assistant_message", "text": "好，记住了。"},
            ],
        )
        built = memory.build_memory(conversation_id)
        self.assertGreaterEqual(built["preferences"], 1)

        context = memory.retrieve_context(conversation_id, "今天深圳天气怎么样？")
        packed = context["cognitive_context"]
        pref_text = " ".join(item["text"] for item in packed["preference_memory"])
        self.assertIn("小弟", pref_text)
        self.assertIn("结论", pref_text)
        evidence = " ".join(item["text"] for item in packed["supporting_evidence"])
        self.assertNotIn("pref_", evidence)
        for item in packed["preference_memory"]:
            self.assertEqual(item["source"], "preference_memory")


if __name__ == "__main__":
    unittest.main()
