---
name: cogmem-memory-guide
description: "Use CogMem to search and reuse the user's past conversations, preferences, episodes, and trajectories. Use this skill when the user refers to prior chats, names, preferences, or history, or when auto-recall returned nothing. Progressive retrieval: cogmem_search returns excerpts + ids; then cogmem_get for original text and cogmem_timeline for an episode sequence. Available tools: cogmem_search, cogmem_get, cogmem_timeline, cogmem_status, cogmem_build."
---

# CogMem memory tools

CogMem is the cognitive-memory adapter for this workspace. It follows the same progressive-retrieval protocol as MemOS: auto-recall and search return **excerpts**, not full memory. Expand a hit only when you need the original text or the ordered events of an episode.

## How memory is provided each turn

- **Automatic recall (hook):** At the start of each turn, the system searches with the user's current message and injects relevant **excerpts** into your context. You do not need to call a tool for that.
- **When that is not enough:** If auto-recall is empty, the user message is long or vague, or the excerpts cannot answer the question, **generate a short 2-5 word query** and call `cogmem_search`.
- **Search is not the last step:** `cogmem_search` returns snippets plus `id` / `episode_id`. Use those ids for follow-up tools.

## Call order

```text
auto-recall (hook, every turn)
  → empty / long query / wrong angle
      → cogmem_search(short query)
            → excerpt looks right, need original text
                → cogmem_get(item_id, kind)
            → hit has episode_id, need what happened in order
                → cogmem_timeline(episode_id)
                      → one memory in the timeline needs full text
                          → cogmem_get(item_id, kind="memory")
```

Do not skip to `cogmem_get` or `cogmem_timeline` without an id from search, auto-recall, or timeline.

## Tools

### cogmem_search

- **What it does:** Search episodes, trajectories, preferences, and recent temporal events. Returns **excerpts** with `id`, `kind`, optional `episode_id`, `sufficient`, and `next_tools`.
- **When to call:**
  - Auto-recall did not run or returned nothing.
  - The user query is long or unclear — generate a short query yourself.
  - You need a different angle than auto-recall.
- **Parameters:**
  - `query` (required): short natural-language query (2-5 words).
  - `search_all` (optional): search every stored conversation.
  - `limit` (optional): max excerpts.

### cogmem_get

- **What it does:** Fetch the full original body of one item.
- **When to call:** A search/timeline hit looks relevant but you need the complete content, not the excerpt.
- **Parameters:**
  - `item_id` (required): id from a search hit.
  - `kind` (optional): `memory` | `episode` | `trajectory` | `preference`.

### cogmem_timeline

- **What it does:** Return memory objects inside one episode, in order.
- **When to call:** A search hit has `episode_id` (or `kind=episode`) and you need the surrounding sequence.
- **Parameters:**
  - `episode_id` (required): from the search hit.
  - `limit` (optional).

### cogmem_status

Show counts for the current conversation, or list stored conversation ids. Call this if search keeps returning empty and you need to check whether memory was built.

### cogmem_build

Trigger offline construction (temporal → long-term memory). Prefer this only if auto-build has not run after a long session.

## Quick decision flow

1. **No excerpts in context, or auto-recall said it found nothing**
   → `cogmem_search(query="...")` with a self-generated short query.

2. **Need the original text of a search hit**
   → `cogmem_get(item_id="...", kind="memory|episode|trajectory|preference")`.

3. **Need the ordered events of that episode**
   → `cogmem_timeline(episode_id="...")`.

4. **A timeline memory needs full original text**
   → `cogmem_get(item_id="<memory_id>", kind="memory")`.

5. **Search still empty after a short query**
   → retry `cogmem_search` with `search_all=true`, or call `cogmem_status`.

6. **Long session and status shows temporal events but no episodes**
   → `cogmem_build`.

## Writing good search queries

- Prefer short, focused queries (a few words or one clear question).
- Use concrete terms: names, cities, tools, decisions.
- If the user's message is long, derive one or two sub-queries rather than pasting the whole message.
- Do not treat auto-recall excerpts as complete memory; expand with get/timeline when details matter.
