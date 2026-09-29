import { getJson, postJson } from "./client.js";

const TOOL_NAMES = [
  "cogmem_search",
  "cogmem_get",
  "cogmem_timeline",
  "cogmem_status",
  "cogmem_build",
];

function stringSchema(description, extra = {}) {
  return { type: "string", description, ...extra };
}

function integerSchema(description, extra = {}) {
  return { type: "integer", description, ...extra };
}

function booleanSchema(description) {
  return { type: "boolean", description };
}

function objectSchema(properties, required = []) {
  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function textResult(details, text) {
  return {
    content: [{ type: "text", text }],
    details,
  };
}

function clip(text, limit = 180) {
  const value = String(text || "").trim();
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, limit - 1)}…`;
}

function conversationIdOf(ctx = {}, params = {}) {
  return String(
    params.conversation_id
      || ctx.sessionId
      || ctx.sessionKey
      || ctx.agentId
      || "",
  ).trim();
}

function followUpLine(hit) {
  const calls = [];
  const kind = hit.kind || "memory";
  const id = hit.id || "";
  const episodeId = hit.episode_id || (kind === "episode" ? id : "");
  if (id) {
    calls.push(`cogmem_get(item_id="${id}", kind="${kind}")`);
  }
  if (episodeId) {
    calls.push(`cogmem_timeline(episode_id="${episodeId}")`);
  }
  return calls.length ? `   next: ${calls.join(" ; ")}` : "";
}

function formatHits(body) {
  const hits = Array.isArray(body?.hits) ? body.hits : [];
  const sufficient = Boolean(body?.sufficient);
  const nextTools = Array.isArray(body?.next_tools) ? body.next_tools : [];
  if (!hits.length) {
    return [
      "No CogMem excerpts found (sufficient=false).",
      "Generate a short 2-5 word query and call cogmem_search again, or set search_all=true.",
    ].join("\n");
  }
  const header = sufficient
    ? `Found ${hits.length} CogMem excerpts (sufficient=true). These are snippets, not full memory.`
    : `Found ${hits.length} CogMem excerpts (sufficient=false). These are snippets, not full memory.`;
  const lines = hits.map((hit, index) => {
    const id = hit.id || "unknown";
    const kind = hit.kind || hit.slot || "memory";
    const score = Number(hit.score || 0).toFixed(3);
    const episode = hit.episode_id ? ` episode_id=${hit.episode_id}` : "";
    const row = `${index + 1}. [${kind}:${id}] ${clip(hit.snippet, 180)} (score=${score})${episode}`;
    const next = followUpLine(hit);
    return next ? `${row}\n${next}` : row;
  });
  const footer = sufficient
    ? "If you need original text, call cogmem_get. If you need ordered events in an episode, call cogmem_timeline."
    : `These excerpts may be insufficient. Next tools: ${nextTools.join(", ") || "cogmem_get, cogmem_timeline, cogmem_search"}. Prefer cogmem_get for full text and cogmem_timeline for the episode sequence. If the angle is wrong, call cogmem_search with a short 2-5 word query.`;
  return `${header}\n\n${lines.join("\n")}\n\n${footer}`;
}

function formatItem(body) {
  if (!body?.found || !body.item) {
    return `No CogMem item found for id "${body?.id || ""}".`;
  }
  const item = body.item;
  const title = item.title || item.summary || item.content || item.text || body.id;
  const extra = item.summary && item.title ? `\n\n${clip(item.summary, 1200)}` : "";
  const text = item.text && item.text !== title ? `\n\n${clip(item.text, 4000)}` : "";
  return `[${body.kind}:${body.id}] ${clip(title, 400)}${extra}${text}`.trim();
}

function formatTimeline(body) {
  if (!body?.found) {
    return `No CogMem episode found for id "${body?.episode_id || ""}".`;
  }
  const episode = body.episode || {};
  const heading = episode.title || episode.summary || body.episode_id;
  const memories = body.memories || [];
  if (!memories.length) {
    return `Episode ${body.episode_id}: ${clip(heading, 400)}\n\nNo memory objects in this episode.`;
  }
  const lines = memories.map((item, index) => {
    const text = item.text || item.summary || item.memory_id;
    const memoryId = item.memory_id || "";
    return `${index + 1}. [${memoryId}] ${clip(text, 240)}\n   next: cogmem_get(item_id="${memoryId}", kind="memory")`;
  });
  return `Episode ${body.episode_id}: ${clip(heading, 400)}\n\n${lines.join("\n")}\n\nThese are ordered memories in the episode. Call cogmem_get for a full original item.`;
}

function formatStatus(body) {
  if (body?.conversation_id) {
    return [
      `conversation_id=${body.conversation_id}`,
      `raw_events=${body.raw_events ?? 0}`,
      `temporal_events=${body.temporal_events ?? 0}`,
      `memory_objects=${body.memory_objects ?? 0}`,
      `episodes=${body.episodes ?? 0}`,
      `trajectories=${body.trajectories ?? 0}`,
      `preferences=${body.preferences ?? 0}`,
    ].join("\n");
  }
  const ids = body?.conversation_ids || [];
  return `store conversations=${body?.conversation_count ?? ids.length}\n${ids.slice(0, 20).join("\n")}`.trim();
}

export function registerCogMemTools(api, cfg) {
  const sidecarUrl = String(cfg.sidecarUrl || "").replace(/\/$/, "");
  const timeoutMs = cfg.timeoutMs;

  api.registerTool(
    (ctx) => ({
      name: "cogmem_search",
      label: "CogMem Search",
      description:
        "Search CogMem for past conversations, preferences, episodes, and trajectories. Returns excerpts with id and optional episode_id for follow-up — not full memory. Use when auto-recall returned nothing, the user query is long or unclear (generate a short 2-5 word query yourself), or you need a different search angle. After hits: call cogmem_get(item_id) for original text, cogmem_timeline(episode_id) for ordered events in that episode.",
      parameters: objectSchema(
        {
          query: stringSchema("Natural-language search query. Prefer 2-5 keywords; do not paste the whole user message."),
          conversation_id: stringSchema("Optional CogMem conversation id. Defaults to the current OpenClaw session."),
          search_all: booleanSchema("Search across all stored conversations when the current session has few hits."),
          limit: integerSchema("Maximum number of excerpts to return.", { minimum: 1, maximum: 20 }),
        },
        ["query"],
      ),
      async execute(_toolCallId, params) {
        const query = String(params.query || "").trim();
        const conversationId = conversationIdOf(ctx, params) || "openclaw_unknown";
        const body = await postJson(
          `${sidecarUrl}/v1/openclaw/recall`,
          {
            conversation_id: conversationId,
            query,
            search_all: Boolean(params.search_all),
            limit: Number(params.limit || 8),
          },
          timeoutMs,
        );
        const hits = Array.isArray(body?.hits) ? body.hits : [];
        return textResult(
          {
            conversation_id: body?.conversation_id || conversationId,
            query,
            hits,
            sufficient: Boolean(body?.sufficient),
            next_tools: body?.next_tools || [],
          },
          formatHits(body),
        );
      },
    }),
    { name: "cogmem_search" },
  );

  api.registerTool(
    (_ctx) => ({
      name: "cogmem_get",
      label: "CogMem Get",
      description:
        "Retrieve the full original text of one CogMem item. Use after cogmem_search or cogmem_timeline when an excerpt looks relevant but you need the complete content, not the snippet. Pass item_id and kind from a search hit (memory | episode | trajectory | preference).",
      parameters: objectSchema(
        {
          item_id: stringSchema("Memory, episode, trajectory, or preference id from a cogmem_search hit."),
          kind: stringSchema("Optional kind from the search hit: memory | episode | trajectory | preference."),
          conversation_id: stringSchema("Optional conversation id. Omit to search all stored conversations."),
        },
        ["item_id"],
      ),
      async execute(_toolCallId, params) {
        const body = await postJson(
          `${sidecarUrl}/v1/openclaw/get`,
          {
            item_id: String(params.item_id || "").trim(),
            kind: params.kind || null,
            conversation_id: String(params.conversation_id || "").trim() || null,
          },
          timeoutMs,
        );
        return textResult(body, formatItem(body));
      },
    }),
    { name: "cogmem_get" },
  );

  api.registerTool(
    (_ctx) => ({
      name: "cogmem_timeline",
      label: "CogMem Timeline",
      description:
        "Return the ordered memory objects inside one CogMem episode. Use after cogmem_search when a hit has episode_id and you need the surrounding sequence of what happened. Then call cogmem_get on a memory_id for the original text.",
      parameters: objectSchema(
        {
          episode_id: stringSchema("Episode id from a cogmem_search hit (episode_id field, or id when kind=episode)."),
          conversation_id: stringSchema("Optional conversation id. Omit to search all stored conversations."),
          limit: integerSchema("Maximum number of memory objects to return.", { minimum: 1, maximum: 100 }),
        },
        ["episode_id"],
      ),
      async execute(_toolCallId, params) {
        const body = await postJson(
          `${sidecarUrl}/v1/openclaw/timeline`,
          {
            episode_id: String(params.episode_id || "").trim(),
            conversation_id: String(params.conversation_id || "").trim() || null,
            limit: Number(params.limit || 20),
          },
          timeoutMs,
        );
        return textResult(body, formatTimeline(body));
      },
    }),
    { name: "cogmem_timeline" },
  );

  api.registerTool(
    (ctx) => ({
      name: "cogmem_status",
      label: "CogMem Status",
      description: "Show CogMem store counts for the current conversation, or list stored conversation ids. Use to check whether memory has been built before searching.",
      parameters: objectSchema({
        conversation_id: stringSchema("Optional conversation id. Omit to use the current OpenClaw session."),
        list_all: booleanSchema("List every stored conversation instead of one session."),
      }),
      async execute(_toolCallId, params) {
        const conversationId = params.list_all ? "" : conversationIdOf(ctx, params);
        const query = conversationId ? `?conversation_id=${encodeURIComponent(conversationId)}` : "";
        const body = await getJson(`${sidecarUrl}/v1/status${query}`, timeoutMs);
        return textResult(body, formatStatus(body));
      },
    }),
    { name: "cogmem_status" },
  );

  api.registerTool(
    (ctx) => ({
      name: "cogmem_build",
      label: "CogMem Build",
      description:
        "Trigger offline CogMem construction for a conversation: temporal events become memory objects, episodes, trajectories, and preferences. Prefer this only if auto-build has not run after a long session.",
      parameters: objectSchema({
        conversation_id: stringSchema("Optional conversation id. Defaults to the current OpenClaw session."),
        force_rebuild: booleanSchema("Rebuild even if a checkpoint already exists."),
      }),
      async execute(_toolCallId, params) {
        const conversationId = conversationIdOf(ctx, params);
        if (!conversationId) {
          const details = { ok: false, error: "conversation_id is required" };
          return textResult(details, details.error);
        }
        const body = await postJson(
          `${sidecarUrl}/v1/build`,
          {
            conversation_id: conversationId,
            force_rebuild: Boolean(params.force_rebuild),
          },
          timeoutMs,
        );
        return textResult(body, `Built CogMem for ${conversationId}: episodes=${body.episodes ?? 0} trajectories=${body.trajectories ?? 0}`);
      },
    }),
    { name: "cogmem_build" },
  );

  return TOOL_NAMES;
}

export { TOOL_NAMES };
