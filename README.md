# Cognitive-Memory-Plugin [OpenClaw]

**面向智能体的认知记忆框架：从交互事件构建长期记忆，为后续任务提供相关上下文。**

CogMem 将智能体的交互记录组织为 Memory、Episode 和 Trajectory，结合近期事件与用户偏好生成可供模型使用的记忆上下文。项目采用 Python 核心与平台适配层分离的架构，支持直接通过 Python API 使用，也提供基于 HTTP sidecar 的 OpenClaw 插件。

## 功能概览

- **分层记忆构建**：将事件整理为记忆对象、连续经历（Episode）和主题轨迹（Trajectory）。
- **近期与长期记忆召回**：结合近期事件和已构建的长期记忆，进行混合双向召回与上下文编译。
- **用户偏好提取**：从用户话轮中提取称呼、风格、角色和约束，作为行为指导加入上下文。
- **渐进式检索**：先返回摘要与标识，再按需读取全文或经历时间线。
- **OpenClaw 自动接入**：通过生命周期 hook 完成回答前召回、回合结束写入和记忆构建。
- **本地持久化**：使用 JSON / JSONL 文件保存数据；默认记忆处理流程无需 GPU、外部 LLM 或 embedding 服务。

## 架构

```text
OpenClaw Gateway
        │ 生命周期 hook / 记忆工具
        ▼
openclaw-cogmem（平台适配层）
        │ HTTP
        ▼
CogMem sidecar（FastAPI）
        │
        ▼
cogmem-core ◄──────── Python API
  ├── Construction   事件 → Memory → Episode → Trajectory
  ├── Recall         记忆召回、排序与上下文编译
  ├── Adapters       平台事件与核心数据结构转换
  └── Repositories   本地 JSON / JSONL 存储
```

记忆构建与在线召回分别由 `ConstructionService` 和 `RecallService` 负责。平台插件处理事件接入与工具注册，记忆算法由核心实现。

## 环境要求

| 组件 | 要求 |
| --- | --- |
| Python 核心与 sidecar | Python 3.10 或更高版本 |
| Python 运行依赖 | FastAPI、Uvicorn、Pydantic，安装核心时自动安装 |
| OpenClaw 接入（可选） | 插件声明兼容 `openclaw >= 2026.3.0`；需自行安装并配置 Gateway 及其 Node.js 运行环境 |
| LLM 记忆构建（可选） | 提供兼容 Chat Completions 接口的模型服务 |

## 快速开始

以下命令均在下载或克隆后的**仓库根目录**执行。

### 1. 安装 Python 核心

创建虚拟环境：

```bash
python -m venv .venv
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装并检查命令行入口：

```bash
python -m pip install -e ./cogmem-core
cogmem --help
```

### 2. 使用 Python API

以下示例演示事件写入、长期记忆构建和上下文召回，无需启动 OpenClaw 或 HTTP 服务。

```python
from cogmem_core import CogMem

memory = CogMem(store_path="./cogmem_store")
conversation_id = "demo_session"

memory.append_events(
    conversation_id=conversation_id,
    events=[
        {"event_type": "user_message", "text": "帮我查订单状态"},
        {"event_type": "assistant_message", "text": "订单已经发货"},
    ],
)
memory.build_memory(conversation_id)

context = memory.retrieve_context(
    conversation_id=conversation_id,
    query="用户之前的订单状态是什么？",
)
print(context["prompt_text"])
```

`append_events()` 写入事件，`build_memory()` 构建长期记忆，`retrieve_context()` 返回包含 `prompt_text` 的上下文结果。尚未构建的近期事件也可参与召回。

### 3. 启动 HTTP 服务（可选）

接入 OpenClaw 或通过 HTTP 使用 CogMem 时，启动 sidecar：

```bash
cogmem sidecar --store ./cogmem_store --host 127.0.0.1 --port 8787
```

也可使用 Python 模块入口：

```bash
python -m cogmem_core.sidecar --store ./cogmem_store --host 127.0.0.1 --port 8787
```

服务启动后，在浏览器中访问 `http://127.0.0.1:8787/health`，应返回 `ok: true` 和存储目录路径。接口文档位于 `http://127.0.0.1:8787/docs`。

`--store` 指定数据目录；相对路径以启动服务时的工作目录为基准。需要复用已有记忆时，应指向同一存储目录。

## 接入 OpenClaw

### 1. 安装插件

保持 sidecar 运行，在 OpenClaw 所在机器的另一个终端中执行（该环境需已安装 `cogmem-core`）：

```bash
cogmem openclaw-install
```

该命令将 Python 包内附带的插件复制到 `~/.openclaw/plugins/openclaw-cogmem`，并输出安装路径与配置片段。

### 2. 合并配置

将命令输出的配置合并到 `~/.openclaw/openclaw.json`，保留已有配置。结构如下：

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
      "paths": ["<插件安装目录的绝对路径>"]
    }
  }
}
```

将 `load.paths` 中的占位符替换为安装命令输出的绝对路径。Windows 路径可使用 `/`，例如 `C:/Users/your-name/.openclaw/plugins/openclaw-cogmem`。上述 hook 配置用于允许记忆上下文注入与对话访问。

### 3. 重启并验证

```bash
openclaw gateway restart
```

插件成功注册时，Gateway 日志中会出现：

```text
cogmem plugin registered sidecar=http://127.0.0.1:8787
```

随后进行一次对话，检查 sidecar 是否收到 `/v1/openclaw/recall` 与 `/v1/openclaw/ingest` 请求。新存储目录初始为空，历史记忆需要通过后续对话积累或日志导入获得。

### 自动记忆流程

| 触发时机 | 行为 |
| --- | --- |
| `before_prompt_build` | 召回相关记忆，将摘要加入模型上下文 |
| `agent_end` | 写入本轮事件，并按内置计数周期触发构建 |
| `before_compaction` / `session_end` | 启用 `autoBuild` 时构建当前会话的长期记忆 |

默认在当前会话召回命中少于 3 条时尝试跨会话召回。周期构建默认每累计 10 次符合条件的回合结束事件触发一次；计数器属于插件实例，触发时构建该事件对应的会话。

### 插件配置

以下字段可写入 `plugins.entries.cogmem.config`，以仓库中的 `openclaw.plugin.json` 为准：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `sidecarUrl` | `http://127.0.0.1:8787` | 记忆服务地址 |
| `recallConversationId` | `""` | 留空使用当前会话；非空时指定召回会话 |
| `autoRecall` | `true` | 回答前自动召回 |
| `autoCapture` | `true` | 回合结束后自动写入事件 |
| `autoBuild` | `true` | 在压缩上下文前、会话结束时自动构建 |
| `minPromptLength` | `4` | 跳过短于该长度的查询 |
| `timeoutMs` | `8000` | HTTP 请求超时时间，单位为毫秒 |

`buildEveryTurns`、`searchAllFallback` 和 `recallMinHits` 为 `index.js` 中的内置参数，当前配置 schema 不接受这三个字段。周期构建位于 `autoCapture` 流程中，因此仅关闭 `autoBuild` 不会关闭周期构建。

## 记忆工具与 HTTP 接口

| 工具 / 操作 | 功能 | HTTP 接口 |
| --- | --- | --- |
| `cogmem_search` | 返回相关记忆摘要及标识 | `POST /v1/openclaw/recall` |
| `cogmem_get` | 按标识获取完整条目 | `POST /v1/openclaw/get` |
| `cogmem_timeline` | 获取 Episode 内按序排列的记忆 | `POST /v1/openclaw/timeline` |
| `cogmem_status` | 查看会话或全库状态 | `GET /v1/status` |
| `cogmem_build` | 触发长期记忆构建 | `POST /v1/build` |
| 自动写入 | 导入 OpenClaw 事件 | `POST /v1/openclaw/ingest` |

渐进检索流程为：`cogmem_search` 获取摘要及 `id` / `episode_id`，再使用 `cogmem_get` 或 `cogmem_timeline` 补充细节。工具使用说明见 [记忆工具指南](openclaw-cogmem/skill/cogmem-memory-guide/SKILL.md)。

sidecar 还提供通用事件写入、上下文召回和偏好提取接口，具体请求结构可查看运行中的 `/docs` 页面。

## 导入历史会话

运行仓库自带的最小 OpenClaw 日志示例：

```bash
python cogmem-core/examples/ingest_openclaw_log.py
```

或指定自己的 JSONL 会话日志与存储目录：

```bash
python cogmem-core/examples/ingest_openclaw_log.py /path/to/session.jsonl --store ./cogmem_store
```

该脚本直接调用 Python 核心，依次完成事件转换、写入、构建和示例召回，无需启动 sidecar。

## 可选 LLM 构建

默认使用启发式 / 词法判定完成 Episode 切分和 Trajectory 聚类。若需启用 LLM 判定，在启动 Python 程序或 sidecar **之前**设置以下环境变量：

| 环境变量 | 设置值 / 说明 |
| --- | --- |
| `COGMEM_EPISODE_BUILDER` | `locomo_llm`，启用 LLM Episode 切分 |
| `COGMEM_TRAJECTORY_BUILDER` | `locomo_llm`，启用 LLM Trajectory 归类 |
| `COGMEM_LLM_BASE_URL` | 模型服务的 API 基础地址；代码会追加 `/chat/completions` |
| `COGMEM_LLM_API_KEY` | 模型服务的访问密钥 |
| `COGMEM_LLM_MODEL` | 模型名称 |
| `COGMEM_LLM_TIMEOUT` | 可选，请求超时秒数，默认 `120` |

基础地址、密钥、模型名称未配齐时，`locomo_llm` 会回退到默认 `locomo` 构建方式。事件写入与在线召回流程本身不调用 LLM。

## 开发与测试

安装开发依赖并运行现有测试：

```bash
python -m pip install -e "./cogmem-core[dev]"
python -m unittest discover -s cogmem-core/tests -v
```

仓库结构：

```text
.
├── cogmem-core/
│   ├── src/cogmem_core/       Python 核心、sidecar 与随包分发的插件
│   ├── examples/             使用示例
│   ├── scripts/              构建等辅助脚本
│   └── tests/                核心与接口测试
├── openclaw-cogmem/           OpenClaw 插件源码
├── docs/                     架构图与自动化说明
├── LICENSE
└── README.md
```

修改插件时需注意：`cogmem openclaw-install` 使用的是 `cogmem-core/src/cogmem_core/data/openclaw_plugin/` 内的随包副本。发布前应保持其与顶层 `openclaw-cogmem/` 源码一致。

欢迎通过开源托管平台的 Issue 和 Pull Request 反馈问题、补充文档或提交改进。问题报告建议附上运行环境、复现步骤及脱敏日志；代码改动请附上相应验证结果。

## 当前实现范围

- 当前版本为 `0.3.0`，提供本地文件存储和 OpenClaw 适配；其他平台可通过 Python API 或 HTTP 接口集成。
- 默认 Episode / Trajectory 判定与偏好提取依赖启发式规则，召回与构建效果需结合实际数据评估。
- Tool Execution Memory 为可选能力，Python 核心默认关闭。

## 文档

- [Python 核心说明](cogmem-core/README.md)
- [OpenClaw 插件说明](openclaw-cogmem/README.md)
- [自动化流程说明](docs/cogmem_automation_explained_zh.md)
- [OpenClaw 集成审查与待改进项](docs/openclaw_review_2026-09-29.md)
- [项目架构图](docs/cogmem_plugin_architecture.svg)

## 许可证

本项目采用 [MIT License](LICENSE)。
