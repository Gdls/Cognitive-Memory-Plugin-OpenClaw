from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..adapters import OpenClawAdapter
from ..facade import CogMem
from ..recall.progressive import empty_progressive_recall, progressive_recall_fields


class EventIngestRequest(BaseModel):
    conversation_id: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class RecallRequestBody(BaseModel):
    conversation_id: str = ""
    query: str = ""
    include_temporal: bool = True
    include_preferences: bool = True
    include_execution: bool = False
    limit: int = 8
    search_all: bool = False


class BuildRequestBody(BaseModel):
    conversation_id: str
    force_rebuild: bool = False


class PreferenceRequestBody(BaseModel):
    conversation_id: str


class OpenClawIngestRequest(BaseModel):
    payload: Any = None
    path: str | None = None
    conversation_id: str | None = None


class OpenClawGetRequest(BaseModel):
    item_id: str
    conversation_id: str | None = None
    kind: str | None = None


class OpenClawTimelineRequest(BaseModel):
    episode_id: str
    conversation_id: str | None = None
    limit: int = 20


def create_app(store_path: str | Path | None = None) -> FastAPI:
    resolved = Path(
        store_path or os.environ.get("COGMEM_STORE", "./cogmem_store")
    ).expanduser().resolve()
    memory = CogMem(store_path=resolved)
    adapter = OpenClawAdapter()
    app = FastAPI(title="CogMem Sidecar", version="0.1.0")
    app.state.memory = memory
    app.state.adapter = adapter
    app.state.store_path = str(resolved)

    logger = logging.getLogger("cogmem.sidecar")
    log_body = os.environ.get("COGMEM_LOG_BODY", "0") == "1"

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        body_bytes = await request.body() if log_body else b""
        response = await call_next(request)
        if response.status_code >= 400 and request.url.path.startswith("/v1/"):
            if body_bytes:
                try:
                    preview = body_bytes.decode("utf-8", errors="replace")[:500]
                except Exception:
                    preview = "<binary>"
                logger.warning("REQ %s body=%s -> %s", request.url.path, preview, response.status_code)
            else:
                logger.warning("RES %s -> %s", request.url.path, response.status_code)
        elif request.url.path.startswith("/v1/"):
            logger.debug("RES %s -> %s", request.url.path, response.status_code)
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "store": app.state.store_path}

    @app.get("/v1/status")
    def status(conversation_id: str | None = None) -> dict[str, Any]:
        return app.state.memory.get_status(conversation_id)

    @app.post("/v1/events")
    def append_events(body: EventIngestRequest) -> dict[str, Any]:
        if not body.events:
            raise HTTPException(status_code=400, detail="events must not be empty")
        return app.state.memory.append_events(body.conversation_id, body.events)

    @app.post("/v1/recall")
    def recall(body: RecallRequestBody) -> dict[str, Any]:
        return app.state.memory.retrieve_context(
            body.conversation_id,
            body.query,
            include_temporal=body.include_temporal,
            include_preferences=body.include_preferences,
            include_execution=body.include_execution,
            limit=body.limit,
        )

    @app.post("/v1/build")
    def build(body: BuildRequestBody) -> dict[str, Any]:
        return app.state.memory.build_memory(
            body.conversation_id, force_rebuild=body.force_rebuild
        )

    @app.post("/v1/preferences")
    def derive_preferences(body: PreferenceRequestBody) -> dict[str, Any]:
        return app.state.memory.derive_preferences(body.conversation_id)

    @app.post("/v1/openclaw/ingest")
    def ingest_openclaw(body: OpenClawIngestRequest) -> dict[str, Any]:
        payload = body.path or body.payload
        if payload is None:
            raise HTTPException(status_code=400, detail="path or payload is required")
        conversion = app.state.adapter.convert(payload)
        conversation_id = body.conversation_id or conversion.conversation_id
        appended = app.state.memory.append_events(conversation_id, conversion.events)
        return {
            "conversation_id": conversation_id,
            "record_count": conversion.record_count,
            "skipped": conversion.skipped,
            "workspace": conversion.workspace,
            **appended,
        }

    @app.post("/v1/openclaw/recall")
    def recall_openclaw(body: RecallRequestBody) -> dict[str, Any]:
        if not body.query or not body.conversation_id:
            logger.warning(
                "recall_openclaw skipped: conversation_id=%r query=%r",
                body.conversation_id,
                body.query,
            )
            return empty_progressive_recall(body.conversation_id or "", body.query or "")
        conversation_ids = [body.conversation_id]
        store = app.state.memory.store
        list_ids = getattr(store, "list_conversation_ids", None)
        if body.search_all and callable(list_ids):
            for conversation_id in list_ids():
                if conversation_id not in conversation_ids:
                    conversation_ids.append(conversation_id)
        context = None
        best_hits = -1
        for conversation_id in conversation_ids:
            candidate = app.state.memory.retrieve_context(
                conversation_id,
                body.query,
                include_temporal=body.include_temporal,
                include_preferences=body.include_preferences,
                include_execution=body.include_execution,
                limit=body.limit,
            )
            hits = _recall_hit_count(candidate)
            if context is None or hits > best_hits:
                context = candidate
                best_hits = hits
        assert context is not None
        progressive = progressive_recall_fields(context, query=body.query, limit=12)
        return {
            **context,
            **progressive,
        }

    @app.post("/v1/openclaw/get")
    def get_openclaw(body: OpenClawGetRequest) -> dict[str, Any]:
        return app.state.memory.get_item(
            body.item_id,
            conversation_id=body.conversation_id,
            kind=body.kind,
        )

    @app.post("/v1/openclaw/timeline")
    def timeline_openclaw(body: OpenClawTimelineRequest) -> dict[str, Any]:
        return app.state.memory.get_timeline(
            body.episode_id,
            conversation_id=body.conversation_id,
            limit=body.limit,
        )

    return app


def _recall_hit_count(context: dict[str, Any]) -> int:
    packed = context.get("cognitive_context") or {}
    diagnostics = packed.get("diagnostics") or {}
    long_term = int(diagnostics.get("long_term_candidates") or 0)
    episodes = int(diagnostics.get("episode_candidates") or 0)
    trajectories = int(diagnostics.get("trajectory_candidates") or 0)
    evidence = len(packed.get("supporting_evidence") or [])
    background = len(packed.get("long_term_background") or [])
    situation = len(packed.get("current_situation") or [])
    constructed = long_term + episodes + trajectories
    # Prefer conversations that already have built episodes/memories over
    # a current session whose temporal log only repeats the query.
    return constructed * 1000 + background * 10 + situation * 10 + evidence
