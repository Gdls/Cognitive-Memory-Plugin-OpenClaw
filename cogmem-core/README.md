# CogMem Core

Python 记忆内核。接到 OpenClaw 的完整步骤见仓库根目录 [README.md](../README.md)。

## 从源码安装

在仓库根目录：

```bash
pip install -e ./cogmem-core
```

或在本目录：

```bash
pip install -e .
```

需要 Python 3.10+。装完：

```bash
cogmem sidecar --store ./cogmem_store --host 127.0.0.1 --port 8787
cogmem openclaw-install
```

OpenClaw 插件随包装在 `cogmem_core/data/openclaw_plugin/`。`openclaw-install` 会拷到 `~/.openclaw/plugins/openclaw-cogmem`，并打印 `openclaw.json` 片段。Gateway 仍需 Node。

也可以打成 wheel 再装：

```bash
python scripts/build_wheel.py
pip install dist/cogmem_core-0.3.0-py3-none-any.whl
```

```python
from cogmem_core import CogMem
```

## Source layout

```text
src/cogmem_core/
  contracts/       Stable event, build, memory, and recall data contracts
  construction/    Offline or manually triggered cognitive-memory building
  recall/          Online retrieval, ranking, and context assembly
  adapters/        Agent-platform payload translation boundary
                   (OpenClaw JSONL <-> CogMemEvent / ContextPack)
  sidecar/         HTTP surface used by platform adapters (recall/get/timeline)
  repositories/    Storage protocols and local implementations
  facade.py        Backward-compatible public CogMem API
```

The runtime has two separate paths. `ConstructionService` consumes accumulated
agent logs outside the latency-sensitive request path. `RecallService` reads
already-built memories and recent temporal events for real-time agent requests.
Platform adapters depend on the contracts and do not contain memory algorithms.

`build_memory()` runs LoCoMo-style construction:

- **Episodes**: rolling-window segmentation (research `build_rolling_episodes`).
  Default judge is a lexical/heuristic segmenter; set `COGMEM_EPISODE_BUILDER=locomo_llm`
  plus `COGMEM_LLM_*` to use an OpenAI-compatible model.
- **Trajectories**: BM25 + temporal candidate recall, then title/reason cluster
  membership (research `build_title_reason_clusters`). Default judge is lexical;
  `COGMEM_TRAJECTORY_BUILDER=locomo_llm` enables the LLM classifier.

`append_events` still does not call an LLM. `retrieve_context` runs hybrid
bidirectional recall, then compiles a compact Cognitive Context (drop
unrelated slots, compress evidence, keep Preference as guidance only).
Use `ConstructionConfig(episode_builder="placeholder")` to restore the old
chunk builder.

`cogmem-core` 是 CogMem 工程化插件内核的最小项目骨架。它与 LoCoMo 实验代码解耦，目标是先稳定统一 Core API，再通过薄适配层接入 OpenClaw 等平台（hook + 可调用函数），后续再接 HTTP、MCP、LangChain。

当前 v0.3 重点：

- 统一 `CogMem` Core API；
- 支持 Agent Event Log；
- 提供本地 JSONL/JSON 存储；
- 提供 Temporal Memory；
- 提供 Long-Term Background Memory 的最小占位构建；
- 提供 Preference Memory：从用户话轮与 Trajectory 主题抽出称呼、风格、角色和约束，作为行为调节项进入 prompt，不作为事实证据。
- 预留 Tool Execution Memory 接口；
- 输出 Cognitive Context Construction。

## 最小示例

```python
from cogmem_core import CogMem

memory = CogMem(store_path="./cogmem_store")

memory.append_events(
    conversation_id="sess_001",
    events=[
        {"event_type": "user_message", "text": "帮我查订单状态"},
        {"event_type": "assistant_message", "text": "订单已经发货"},
    ],
)

context = memory.retrieve_context(
    conversation_id="sess_001",
    query="用户之前的订单状态是什么？",
)

prompt = memory.format_prompt_context(
    conversation_id="sess_001",
    query="用户之前的订单状态是什么？",
)

print(prompt)
```

## 核心设计

必选 memory views：

```text
Long-Term Background Memory
Preference Memory based on Trajectory
Temporal Memory
```

可选 memory view：

```text
Tool Execution Memory
```

运行时路径：

```text
User Query
  -> Preference Memory
  -> Long-Term Background Memory
  -> Temporal Memory
  -> Optional Tool Execution Memory
  -> Cognitive Context Construction
  -> Agent Prompt
```

## 当前限制

- Episode / Trajectory 已迁入 `construction/`（滚动切分 + BM25 聚类）。无 LLM 时用启发式/词法判定，质量低于研究侧的 LLM 切分与关系分类。
- `retrieve_context` 使用 Hybrid 双向召回，再做 Cognitive Context 编译（相关过滤、证据截断、token 预算）。无 embedding、无 LLM。
- Preference Memory 从用户话轮做启发式抽取（称呼、风格、角色、约束），无 LLM。质量低于人工标注或 LLM 摘要。
