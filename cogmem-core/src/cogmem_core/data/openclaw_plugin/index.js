import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { errorMessage, postJson } from "./client.js";
import { registerCogMemTools } from "./tools.js";

const DEFAULTS = {
  sidecarUrl: "http://127.0.0.1:8787",
  recallConversationId: "",
  autoRecall: true,
  autoCapture: true,
  autoBuild: true,
  buildEveryTurns: 10,
  searchAllFallback: true,
  recallMinHits: 3,
  minPromptLength: 4,
  timeoutMs: 8000,
};

const SEARCH_HINT = [
  "CogMem auto-recall found no relevant memories for this turn.",
  "If the user refers to past conversations, preferences, or history,",
  "call cogmem_search with a short 2-5 word query you generate.",
  "Search returns excerpts only. Then call cogmem_get(item_id) for original text",
  "or cogmem_timeline(episode_id) for ordered events in that episode.",
].join("\n");

function hasRecallHits(body) {
  const hits = Array.isArray(body?.hits) ? body.hits.length : 0;
  if (hits > 0) {
    return hits;
  }
  const ctx = body?.cognitive_context || {};
  const evidence = (ctx.supporting_evidence || []).length;
  const background = (ctx.long_term_background || []).length;
  const situation = (ctx.current_situation || []).length;
  const related = (ctx.related_experience || []).length;
  return evidence + background + situation + related;
}

function registerMemoryCapability(api) {
  if (typeof api.registerMemoryCapability !== "function") {
    return;
  }
  try {
    api.registerMemoryCapability({
      promptBuilder: ({ availableTools } = {}) => {
        const tools = availableTools instanceof Set ? availableTools : new Set();
        const lines = [
          "## Memory (CogMem)",
          "This workspace uses CogMem. Auto-recall injects excerpts before you answer; they are not full memory.",
          "Call order: cogmem_search → cogmem_get (original text) and/or cogmem_timeline (episode sequence).",
        ];
        if (tools.has("cogmem_search")) {
          lines.push("- `cogmem_search` — when auto-recall is empty or the user query is long/vague; pass a short 2-5 word query. Returns excerpts + ids.");
        }
        if (tools.has("cogmem_get")) {
          lines.push("- `cogmem_get` — after search, when an excerpt looks relevant and you need the original text.");
        }
        if (tools.has("cogmem_timeline")) {
          lines.push("- `cogmem_timeline` — after search, when a hit has episode_id and you need ordered events in that episode.");
        }
        if (tools.has("cogmem_status")) {
          lines.push("- `cogmem_status` — show store counts for the current conversation.");
        }
        if (tools.has("cogmem_build")) {
          lines.push("- `cogmem_build` — trigger offline long-term memory construction.");
        }
        if (lines.length === 3) {
          return [];
        }
        lines.push("- Prefer recalled memory over assuming prior context is unavailable.");
        lines.push("");
        return lines;
      },
    });
  } catch (error) {
    api.logger?.warn?.(`cogmem memory capability skipped: ${errorMessage(error)}`);
  }
}

export default definePluginEntry({
  id: "cogmem",
  name: "CogMem",
  description: "Cognitive memory adapter for OpenClaw: auto-recall, capture, and callable memory tools.",
  kind: "memory",
  register(api) {
    const cfg = { ...DEFAULTS, ...(api.pluginConfig || {}) };
    const sidecarUrl = String(cfg.sidecarUrl || DEFAULTS.sidecarUrl).replace(/\/$/, "");
    cfg.sidecarUrl = sidecarUrl;

    registerCogMemTools(api, cfg);
    registerMemoryCapability(api);

    if (cfg.autoRecall) {
      api.on("before_prompt_build", async (event, ctx) => {
        const query = extractQuery(event);
        if (shouldSkipTurn(event, ctx, query)) {
          return;
        }
        if (!query || query.length < Number(cfg.minPromptLength || 4)) {
          return;
        }
        const conversationId = recallConversationIdOf(cfg, event, ctx);
        if (!conversationId) {
          return;
        }
        try {
          let body = await postJson(
            `${sidecarUrl}/v1/openclaw/recall`,
            {
              conversation_id: conversationId,
              query,
              search_all: false,
            },
            cfg.timeoutMs,
          );
          const minHits = Number(cfg.recallMinHits || 3);
          if (cfg.searchAllFallback && hasRecallHits(body) < minHits) {
            const fallback = await postJson(
              `${sidecarUrl}/v1/openclaw/recall`,
              {
                conversation_id: conversationId,
                query,
                search_all: true,
              },
              cfg.timeoutMs,
            );
            if (hasRecallHits(fallback) > hasRecallHits(body)) {
              body = fallback;
            }
          }
          const prependContext = body?.openclaw?.prependContext || "";
          if (!prependContext) {
            return { prependContext: SEARCH_HINT };
          }
          return { prependContext };
        } catch (error) {
          api.logger.warn(`cogmem recall failed: ${errorMessage(error)}`);
        }
      });
    }

    if (cfg.autoCapture) {
      const turnCounter = { count: 0 };
      api.on("agent_end", async (event, ctx) => {
        if (shouldSkipTurn(event, ctx)) {
          return;
        }
        const payload = {
          sessionId: conversationIdOf(event, ctx),
          messages: extractMessages(event),
        };
        if (!payload.messages.length) {
          return;
        }
        try {
          await postJson(
            `${sidecarUrl}/v1/openclaw/ingest`,
            {
              conversation_id: payload.sessionId,
              payload,
            },
            cfg.timeoutMs,
          );
        } catch (error) {
          api.logger.warn(`cogmem ingest failed: ${errorMessage(error)}`);
        }
        turnCounter.count += 1;
        const every = Number(cfg.buildEveryTurns || 0);
        if (every > 0 && turnCounter.count >= every) {
          turnCounter.count = 0;
          try {
            await postJson(
              `${sidecarUrl}/v1/build`,
              { conversation_id: payload.sessionId },
              cfg.timeoutMs,
            );
          } catch (error) {
            api.logger.warn(`cogmem periodic build failed: ${errorMessage(error)}`);
          }
        }
      });
    }

    if (cfg.autoBuild) {
      const buildHandler = async (event, ctx) => {
        try {
          await postJson(
            `${sidecarUrl}/v1/build`,
            { conversation_id: conversationIdOf(event, ctx) },
            cfg.timeoutMs,
          );
        } catch (error) {
          api.logger.warn(`cogmem build failed: ${errorMessage(error)}`);
        }
      };
      api.on("before_compaction", buildHandler);
      api.on("session_end", buildHandler);
    }

    api.logger.info(`cogmem plugin registered sidecar=${sidecarUrl} tools=cogmem_search,cogmem_get,cogmem_timeline,cogmem_status,cogmem_build`);
  },
});

function recallConversationIdOf(cfg = {}, event = {}, ctx = {}) {
  const configured = String(cfg.recallConversationId || "").trim();
  return configured || conversationIdOf(event, ctx);
}

function conversationIdOf(event = {}, ctx = {}) {
  return String(
    event.sessionId
      || event.sessionKey
      || ctx.sessionId
      || ctx.sessionKey
      || event.session?.id
      || "openclaw_unknown",
  );
}

function extractQuery(event = {}) {
  if (typeof event.prompt === "string" && event.prompt.trim()) {
    return unwrapUserText(event.prompt);
  }
  const messages = extractMessages(event);
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "user") {
      return unwrapUserText(textOf(message.content));
    }
  }
  return "";
}

function extractMessages(event = {}) {
  const raw = event.messages || event.sessionMessages || event.preparedMessages || [];
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .map((item) => {
      if (item?.role && item?.content !== undefined) {
        return { role: item.role, content: item.content, timestamp: item.timestamp };
      }
      if (item?.type === "message" && item.message) {
        return item.message;
      }
      return null;
    })
    .filter(Boolean);
}

function shouldSkipTurn(event = {}, ctx = {}, query = null) {
  const trigger = String(ctx.trigger || event.trigger || "");
  if (trigger === "memory" || trigger.includes("memory")) {
    return true;
  }
  const normalized = (query ?? extractQuery(event)).toLowerCase();
  if (!normalized) {
    return true;
  }
  if (normalized.includes("pre-compaction memory flush") || normalized === "no_reply") {
    return true;
  }
  return false;
}

function unwrapUserText(text) {
  const fence = /(?:Conversation info|Sender)\s+\(untrusted metadata\):\s*```json[\s\S]*?```/gi;
  return String(text || "").replace(fence, "").trim();
}

function textOf(content) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : part?.text || ""))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}
