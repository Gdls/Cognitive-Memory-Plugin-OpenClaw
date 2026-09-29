from __future__ import annotations

from pathlib import Path
import shutil
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

from cogmem_core import CogMem
from cogmem_core.sidecar import create_app

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "openclaw" / "mini_session.jsonl"


@unittest.skipIf(TestClient is None, "fastapi is not installed")
class SidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Path(__file__).resolve().parent / ".tmp" / "sidecar"
        if self.store.exists():
            shutil.rmtree(self.store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.client = TestClient(create_app(self.store))

    def tearDown(self) -> None:
        shutil.rmtree(self.store, ignore_errors=True)

    def test_health_and_event_recall_build(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ok"])

        appended = self.client.post(
            "/v1/events",
            json={
                "conversation_id": "sess_http",
                "events": [
                    {"event_type": "user_message", "text": "今天深圳天气怎么样？"},
                    {"event_type": "assistant_message", "text": "我去查深圳天气。"},
                ],
            },
        )
        self.assertEqual(appended.status_code, 200)
        self.assertEqual(appended.json()["appended"], 2)

        built = self.client.post("/v1/build", json={"conversation_id": "sess_http"})
        self.assertEqual(built.status_code, 200)
        self.assertGreaterEqual(built.json()["episodes"], 1)

        recalled = self.client.post(
            "/v1/recall",
            json={"conversation_id": "sess_http", "query": "用户之前问过什么天气？"},
        )
        self.assertEqual(recalled.status_code, 200)
        self.assertIn("深圳天气", recalled.json()["prompt_text"])

        status = self.client.get("/v1/status", params={"conversation_id": "sess_http"})
        self.assertEqual(status.json()["memory_objects"], 2)

    def test_openclaw_ingest_and_recall_returns_prepend_context(self) -> None:
        ingested = self.client.post(
            "/v1/openclaw/ingest",
            json={"path": str(FIXTURE)},
        )
        self.assertEqual(ingested.status_code, 200)
        conversation_id = ingested.json()["conversation_id"]
        self.assertEqual(conversation_id, "sess_openclaw_mini")
        self.client.post("/v1/build", json={"conversation_id": conversation_id})

        recalled = self.client.post(
            "/v1/openclaw/recall",
            json={"conversation_id": conversation_id, "query": "用户之前问过什么天气？"},
        )
        body = recalled.json()
        self.assertIn("prependContext", body["openclaw"])
        self.assertIn("深圳天气", body["openclaw"]["prependContext"])
        self.assertIn("CogMem recalled excerpts", body["openclaw"]["prependContext"])
        self.assertNotIn("Cognitive Context Construction", body["openclaw"]["prependContext"])
        self.assertTrue(body["hits"])
        self.assertIn("sufficient", body)
        self.assertTrue(body["next_tools"])
        self.assertTrue(any(hit.get("kind") == "memory" for hit in body["hits"]))
        self.assertTrue(any(hit.get("episode_id") for hit in body["hits"]))
        self.assertTrue(any("深圳" in str(hit.get("snippet") or "") for hit in body["hits"]))

    def test_openclaw_get_and_timeline_lookup_built_items(self) -> None:
        ingested = self.client.post(
            "/v1/openclaw/ingest",
            json={"path": str(FIXTURE)},
        )
        conversation_id = ingested.json()["conversation_id"]
        self.client.post("/v1/build", json={"conversation_id": conversation_id})

        episodes = CogMem(store_path=self.store).store.load_episodes(conversation_id)
        self.assertTrue(episodes)
        episode = episodes[0]
        episode_id = episode["episode_id"]
        memory_id = (episode.get("memory_ids") or [None])[0]
        self.assertTrue(memory_id)

        got = self.client.post(
            "/v1/openclaw/get",
            json={"item_id": episode_id, "kind": "episode"},
        )
        self.assertEqual(got.status_code, 200)
        self.assertTrue(got.json()["found"])
        self.assertEqual(got.json()["kind"], "episode")

        missing = self.client.post(
            "/v1/openclaw/get",
            json={"item_id": "does_not_exist"},
        )
        self.assertFalse(missing.json()["found"])

        timeline = self.client.post(
            "/v1/openclaw/timeline",
            json={"episode_id": episode_id},
        )
        self.assertTrue(timeline.json()["found"])
        self.assertGreaterEqual(len(timeline.json()["memories"]), 1)
        self.assertEqual(timeline.json()["memories"][0]["memory_id"], memory_id)

    def test_openclaw_recall_search_all_uses_other_conversation(self) -> None:
        ingested = self.client.post(
            "/v1/openclaw/ingest",
            json={"path": str(FIXTURE)},
        )
        conversation_id = ingested.json()["conversation_id"]
        self.client.post("/v1/build", json={"conversation_id": conversation_id})

        recalled = self.client.post(
            "/v1/openclaw/recall",
            json={
                "conversation_id": "tray_session_new",
                "query": "用户之前问过什么天气？",
                "search_all": True,
            },
        )
        body = recalled.json()
        self.assertEqual(body["conversation_id"], conversation_id)
        self.assertIn("深圳天气", body["openclaw"]["prependContext"])
        self.assertIn("sufficient", body)

    def test_openclaw_recall_without_query_returns_search_hint(self) -> None:
        recalled = self.client.post(
            "/v1/openclaw/recall",
            json={"conversation_id": "sess_empty", "query": ""},
        )
        body = recalled.json()
        self.assertEqual(recalled.status_code, 200)
        self.assertFalse(body["sufficient"])
        self.assertEqual(body["next_tools"], ["cogmem_search"])
        self.assertIn("cogmem_search", body["openclaw"]["prependContext"])


if __name__ == "__main__":
    unittest.main()
