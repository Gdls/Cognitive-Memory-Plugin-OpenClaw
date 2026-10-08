# Cognitive-Memory-Plugin [OpenClaw]

**English** | [简体中文](README.zh-CN.md)

**A cognitive memory framework for agents: build long-term memory from interaction events and retrieve relevant context for future tasks.**

CogMem organizes agent interactions into Memory, Episode, and Trajectory structures. It combines recent events and user preferences to assemble memory context for language models. The Python core is separate from platform adapters: use it directly through the Python API or connect OpenClaw through an HTTP sidecar.

## Features

- **Hierarchical memory construction**: organize events into memory objects, coherent episodes, and thematic trajectories.
- **Recent and long-term recall**: combine recent events with constructed memories through hybrid bidirectional retrieval and context compilation.
- **User preferences**: extract names, response styles, roles, and constraints from user turns and include them as behavioral guidance.
- **Progressive retrieval**: retrieve excerpts and identifiers first, then expand individual items or episode timelines as needed.
- **OpenClaw integration**: use lifecycle hooks for recall before responses, event capture after turns, and memory construction.
- **Local persistence**: store data in JSON / JSONL files. Default memory processing requires no GPU, external LLM, or embedding service.

## Architecture

```text
OpenClaw Gateway
        │ Lifecycle hooks / memory tools
        ▼
openclaw-cogmem (platform adapter)
        │ HTTP
        ▼
CogMem sidecar (FastAPI)
        │
        ▼
cogmem-core ◄──────── Python API
  ├── Construction   Events → Memory → Episode → Trajectory
  ├── Recall         Retrieval, ranking, and context compilation
  ├── Adapters       Platform events ↔ core data structures
  └── Repositories   Local JSON / JSONL storage
```

`ConstructionService` handles memory construction, while `RecallService` handles online retrieval. The platform plugin captures events and registers tools; memory algorithms live in the core.

## Requirements

| Component | Requirement |
| --- | --- |
| Python core and sidecar | Python 3.10 or later |
| Python runtime dependencies | FastAPI, Uvicorn, and Pydantic; installed with the core |
| OpenClaw integration (optional) | The plugin declares `openclaw >= 2026.3.0`; install and configure the Gateway and its Node.js runtime separately |
| LLM-assisted construction (optional) | A model service with a compatible Chat Completions interface |

## Quick start

Run the following commands from the **repository root** after downloading or cloning the project.

### 1. Install the Python core

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux / macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the package and check the CLI:

```bash
python -m pip install -e ./cogmem-core
cogmem --help
```

### 2. Use the Python API

This example writes events, constructs long-term memory, and retrieves context. OpenClaw and the HTTP service are not required.

```python
from cogmem_core import CogMem

memory = CogMem(store_path="./cogmem_store")
conversation_id = "demo_session"

memory.append_events(
    conversation_id=conversation_id,
    events=[
        {"event_type": "user_message", "text": "Please check my order status."},
        {"event_type": "assistant_message", "text": "Your order has shipped."},
    ],
)
memory.build_memory(conversation_id)

context = memory.retrieve_context(
    conversation_id=conversation_id,
    query="What was the user's previous order status?",
)
print(context["prompt_text"])
```

`append_events()` stores events, `build_memory()` constructs long-term memory, and `retrieve_context()` returns context containing `prompt_text`. Recent events can also participate in recall before long-term construction runs.

### 3. Start the HTTP service (optional)

Start the sidecar to connect OpenClaw or access CogMem over HTTP:

```bash
cogmem sidecar --store ./cogmem_store --host 127.0.0.1 --port 8787
```

Alternatively, use the Python module entry point:

```bash
python -m cogmem_core.sidecar --store ./cogmem_store --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787/health` in a browser. The response should contain `ok: true` and the storage path. Interactive API documentation is available at `http://127.0.0.1:8787/docs`.

`--store` specifies the data directory. Relative paths are resolved from the working directory used to launch the service. Point to the same directory to reuse existing memories.

## Connect OpenClaw

### 1. Install the plugin

Keep the sidecar running. In another terminal on the OpenClaw machine, with `cogmem-core` installed, run:

```bash
cogmem openclaw-install
```

This copies the plugin bundled in the Python package to `~/.openclaw/plugins/openclaw-cogmem` and prints its installation path and a configuration snippet.

### 2. Merge the configuration

Merge the generated snippet into `~/.openclaw/openclaw.json`, preserving existing settings. The configuration has this structure:

```json
{
  "plugins": {
    "entries": {
      "cogmem": {
        "enabled": true,
        "config": {
          "sidecarUrl": "http://127.0.0.1:8787",
          "autoRecall": true,
          "autoCapture": true,
          "autoBuild": true
        },
        "hooks": {
          "allowPromptInjection": true,
          "allowConversationAccess": true
        }
      }
    },
    "load": {
      "paths": ["<absolute-path-to-installed-plugin>"]
    }
  }
}
```

Replace the placeholder in `load.paths` with the absolute path printed by the installer. Windows paths can use `/`, for example `C:/Users/your-name/.openclaw/plugins/openclaw-cogmem`. The hook settings enable memory context injection and conversation access.

### 3. Restart and verify

```bash
openclaw gateway restart
```

When registration succeeds, the Gateway log includes:

```text
cogmem plugin registered sidecar=http://127.0.0.1:8787
```

Start a conversation and check that the sidecar receives `/v1/openclaw/recall` and `/v1/openclaw/ingest` requests. A new store is empty; memories accumulate through subsequent conversations or log imports.

### Automatic memory lifecycle

| Trigger | Behavior |
| --- | --- |
| `before_prompt_build` | Retrieve relevant memories and add excerpts to model context |
| `agent_end` | Capture events and trigger construction according to the built-in counter |
| `before_compaction` / `session_end` | Construct long-term memory for the current conversation when `autoBuild` is enabled |

By default, the plugin attempts cross-conversation recall when the current conversation returns fewer than 3 hits. Periodic construction runs after every 10 eligible turn-end events. The counter belongs to the plugin instance; construction targets the conversation associated with the triggering event.

### Plugin configuration

The following fields are accepted in `plugins.entries.cogmem.config`, as defined in `openclaw.plugin.json`:

| Setting | Default | Description |
| --- | --- | --- |
| `sidecarUrl` | `http://127.0.0.1:8787` | Memory service URL |
| `recallConversationId` | `""` | Empty uses the current conversation; otherwise specifies a conversation for recall |
| `autoRecall` | `true` | Recall memories before responses |
| `autoCapture` | `true` | Capture events after turns |
| `autoBuild` | `true` | Construct memory before compaction and at session end |
| `minPromptLength` | `4` | Skip queries shorter than this length |
| `timeoutMs` | `8000` | HTTP request timeout in milliseconds |

`buildEveryTurns`, `searchAllFallback`, and `recallMinHits` are built-in parameters in `index.js`; the current configuration schema does not accept these fields. Periodic construction is part of the `autoCapture` flow, so disabling `autoBuild` alone does not disable periodic construction.

## Memory tools and HTTP endpoints

| Tool / operation | Purpose | HTTP endpoint |
| --- | --- | --- |
| `cogmem_search` | Return relevant excerpts and identifiers | `POST /v1/openclaw/recall` |
| `cogmem_get` | Retrieve a stored item by identifier | `POST /v1/openclaw/get` |
| `cogmem_timeline` | Retrieve ordered memories within an Episode | `POST /v1/openclaw/timeline` |
| `cogmem_status` | Inspect conversation or store status | `GET /v1/status` |
| `cogmem_build` | Trigger long-term memory construction | `POST /v1/build` |
| Automatic capture | Import OpenClaw events | `POST /v1/openclaw/ingest` |

Progressive retrieval starts with `cogmem_search` for excerpts and `id` / `episode_id`, followed by `cogmem_get` or `cogmem_timeline` for additional detail. See the [memory tool guide](openclaw-cogmem/skill/cogmem-memory-guide/SKILL.md).

The sidecar also provides generic event ingestion, context retrieval, and preference extraction endpoints. See the running service's `/docs` page for request schemas.

## Import conversation history

Run the bundled minimal OpenClaw log example:

```bash
python cogmem-core/examples/ingest_openclaw_log.py
```

Or specify your own JSONL session log and storage directory:

```bash
python cogmem-core/examples/ingest_openclaw_log.py /path/to/session.jsonl --store ./cogmem_store
```

The script calls the Python core directly to convert events, ingest them, construct memory, and run an example query. The sidecar does not need to be running.

## Optional LLM-assisted construction

By default, Episode segmentation and Trajectory clustering use heuristic / lexical judgments. To enable LLM judgments, set the following environment variables **before** starting the Python application or sidecar:

| Environment variable | Value / description |
| --- | --- |
| `COGMEM_EPISODE_BUILDER` | `locomo_llm` to enable LLM-assisted Episode segmentation |
| `COGMEM_TRAJECTORY_BUILDER` | `locomo_llm` to enable LLM-assisted Trajectory classification |
| `COGMEM_LLM_BASE_URL` | Model service API base URL; the client appends `/chat/completions` |
| `COGMEM_LLM_API_KEY` | Model service API key |
| `COGMEM_LLM_MODEL` | Model name |
| `COGMEM_LLM_TIMEOUT` | Optional request timeout in seconds; defaults to `120` |

If the base URL, API key, or model name is missing, `locomo_llm` falls back to the default `locomo` implementation. Event ingestion and online recall do not themselves call an LLM.

## Development and testing

Install development dependencies and run the existing tests:

```bash
python -m pip install -e "./cogmem-core[dev]"
python -m unittest discover -s cogmem-core/tests -v
```

Repository layout:

```text
.
├── cogmem-core/
│   ├── src/cogmem_core/       Python core, sidecar, and bundled plugin
│   ├── examples/             Usage examples
│   ├── scripts/              Build and maintenance scripts
│   └── tests/                Core and API tests
├── openclaw-cogmem/           OpenClaw plugin source
├── docs/                     Architecture diagrams and lifecycle documentation
├── LICENSE
├── README.md                 English documentation (default)
└── README.zh-CN.md           Chinese documentation
```

When modifying the plugin, note that `cogmem openclaw-install` uses the bundled copy in `cogmem-core/src/cogmem_core/data/openclaw_plugin/`. Keep it synchronized with the top-level `openclaw-cogmem/` source before publishing.

Issues and pull requests are welcome for bug reports, documentation, and improvements. Include your environment, reproduction steps, and sanitized logs in bug reports, and describe validation performed for code changes.

## Current scope

- Version `0.3.0` provides local file storage and an OpenClaw adapter. Other platforms can integrate through the Python API or HTTP endpoints.
- Default Episode / Trajectory judgments and preference extraction rely on heuristics. Evaluate construction and retrieval quality with your own data.
- Tool Execution Memory is optional and disabled by default in the Python core.

## Documentation

Supporting documents currently include Chinese or mixed Chinese/English content.

- [Python core](cogmem-core/README.md)
- [OpenClaw plugin](openclaw-cogmem/README.md)
- [Automatic memory lifecycle](docs/cogmem_automation_explained_zh.md)
- [OpenClaw integration review and known issues](docs/openclaw_review_2026-09-29.md)
- [Architecture diagram](docs/cogmem_plugin_architecture.svg)

## License

This project is licensed under the [MIT License](LICENSE).
