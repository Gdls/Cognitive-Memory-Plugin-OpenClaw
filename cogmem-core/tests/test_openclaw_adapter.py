from __future__ import annotations

from pathlib import Path
import shutil
import unittest

from cogmem_core import CogMem, OpenClawAdapter

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "openclaw" / "mini_session.jsonl"
REPO_LOG_SAMPLES = Path(__file__).resolve().parents[2] / "log_samples"


class OpenClawAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = OpenClawAdapter()

    def test_jsonl_session_maps_to_clean_cogmem_events(self) -> None:
        conversion = self.adapter.convert(FIXTURE)

        self.assertEqual(conversion.conversation_id, "sess_openclaw_mini")
        self.assertEqual(conversion.workspace, "/home/user/.openclaw/workspace")

        texts = [event.text for event in conversion.events]
        types = [event.event_type for event in conversion.events]

        self.assertIn("今天深圳天气怎么样？", texts)
        self.assertIn("你能帮我做个ppt对openclaw进行科普吗？", texts)
        self.assertTrue(any(event.speaker == "Xin Liu" for event in conversion.events))
        self.assertIn("tool_call", types)
        self.assertIn("tool_observation", types)

        self.assertFalse(any("untrusted metadata" in text for text in texts))
        self.assertFalse(any("Pre-compaction memory flush" in text for text in texts))
        self.assertFalse(any(text == "NO_REPLY" for text in texts))
        self.assertFalse(any("VERY_LONG_BASE64" in text for text in texts))
        self.assertFalse(any("EXAMPLESECRETVALUE" in (event.text or "") for event in conversion.events))
        self.assertFalse(
            any(
                "EXAMPLESECRETVALUE" in str(event.arguments or {})
                for event in conversion.events
            )
        )

    def test_duplicate_assistant_text_after_tool_use_is_dropped(self) -> None:
        conversion = self.adapter.convert(FIXTURE)
        assistant = [
            event.text
            for event in conversion.events
            if event.event_type == "assistant_message"
        ]
        self.assertEqual(assistant.count("查询一下今天深圳的天气吧。"), 1)

    def test_live_payload_without_transcript_wrapper_is_accepted(self) -> None:
        events = self.adapter.to_events(
            {
                "sessionId": "live_001",
                "messages": [
                    {"role": "user", "content": "记住我叫张三", "timestamp": "2026-03-15T03:04:51Z"},
                    {"role": "assistant", "content": "好的，张三。"},
                ],
            }
        )
        self.assertEqual(events[0].event_type, "user_message")
        self.assertEqual(events[0].text, "记住我叫张三")
        self.assertEqual(events[1].event_type, "assistant_message")

    def test_from_context_pack_uses_openclaw_prepend_field(self) -> None:
        from cogmem_core.contracts.recall import CognitiveContext, ContextPack

        context = ContextPack(
            conversation_id="sess_openclaw_mini",
            query="深圳天气",
            cognitive_context=CognitiveContext(
                query="深圳天气", conversation_id="sess_openclaw_mini"
            ),
            prompt_text="Cognitive Context Construction\n深圳天气",
        )
        payload = self.adapter.from_context_pack(context)
        self.assertEqual(payload["conversationId"], "sess_openclaw_mini")
        self.assertIn("prependContext", payload)
        self.assertIn("Cognitive Context Construction", payload["prependContext"])

    def test_mini_session_can_be_ingested_and_recalled(self) -> None:
        store = self._temp_store()
        memory = CogMem(store_path=store)
        conversion = self.adapter.convert(FIXTURE)
        memory.append_events(conversion.conversation_id, conversion.events)
        memory.build_memory(conversion.conversation_id)

        prompt = memory.format_prompt_context(conversion.conversation_id, "深圳天气")
        self.assertIn("Cognitive Context Construction", prompt)
        self.assertIn("深圳", prompt)

        identity = memory.format_prompt_context(conversion.conversation_id, "小弟")
        self.assertIn("小弟", identity)

    def test_real_log_samples_unwrap_and_keep_known_facts(self) -> None:
        main_log = REPO_LOG_SAMPLES / "openclaw_main_session_history_log.jsonl"
        zekun_log = REPO_LOG_SAMPLES / "openclaw_zekun_session_history_log.jsonl"
        if not main_log.exists() or not zekun_log.exists():
            self.skipTest("repository log_samples are not available")

        main = self.adapter.convert(main_log)
        self.assertEqual(main.conversation_id, "eb4ae1dd-1eb5-4750-8676-48b8fa7e3f46")
        main_texts = "\n".join(event.text for event in main.events)
        self.assertIn("今天深圳天气怎么样？", main_texts)
        self.assertNotIn("untrusted metadata", main_texts)
        self.assertNotIn("Pre-compaction memory flush", main_texts)
        self.assertGreater(len(main.events), 20)
        self.assertGreater(main.record_count, 20)

        zekun = self.adapter.convert(zekun_log)
        self.assertEqual(zekun.conversation_id, "4ae3d96c-6cc2-41b6-b742-a774c6778933")
        zekun_texts = "\n".join(event.text for event in zekun.events)
        self.assertIn("你叫小弟", zekun_texts)
        self.assertNotIn("tvly-dev-", zekun_texts)
        self.assertFalse(any("data" in str(event.result or "") and len(str(event.result)) > 2000 for event in zekun.events))

    def _temp_store(self) -> Path:
        path = Path(__file__).resolve().parent / ".tmp" / "openclaw_adapter"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path


if __name__ == "__main__":
    unittest.main()
