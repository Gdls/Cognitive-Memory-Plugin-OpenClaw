# OpenClaw CogMem 插件

从源码安装整条链路（核心 + sidecar + 本插件）见仓库根目录 [README.md](../README.md)。

OpenClaw Gateway 的薄适配层：不跑记忆算法，只把平台事件翻译成 CogMem 调用。
职责与 MemOS 的 OpenClaw adapter 同构——**一个算法核心，一层平台适配**：

- 生命周期 hook：自动召回 / 写入 / 构建
- 可调用函数：MemOS 同构的渐进检索（search 只返回 excerpt + id，再 get / timeline）

```text
用户发消息
  -> before_prompt_build  POST /v1/openclaw/recall  -> 注入 excerpts（不是全文）
  -> 命中为空 / 不够时提示模型调用 cogmem_search
回合中（渐进检索，对齐 MemOS）
  -> cogmem_search  返回 excerpt + id + episode_id + sufficient + next_tools
  -> cogmem_get     按 id 取原文
  -> cogmem_timeline 按 episode_id 取有序事件
  -> cogmem_status / cogmem_build
回合结束
  -> agent_end            POST /v1/openclaw/ingest  -> 写入 temporal events
                                        每 10 轮  -> POST /v1/build  -> 构建长期记忆
compaction / session 结束
  -> before_compaction / session_end  POST /v1/build
```

## 架构

```
OpenClaw Gateway (WSL/Node)
  └── cogmem 适配层 (薄 JS plugin)
        ├── registerMemoryCapability  → 告诉模型何时调用记忆函数
        ├── registerTool
        │     cogmem_search / cogmem_get / cogmem_timeline
        │     cogmem_status / cogmem_build
        ├── before_prompt_build       → auto-recall + search_all fallback
        ├── agent_end                 → ingest + 每10轮 build
        └── before_compaction/session_end → build
              └── cogmem-sidecar (Python/FastAPI)
                    └── cogmem-core（算法核心，不含平台逻辑）
                          ├── OpenClawAdapter  → JSONL/事件 → CogMemEvent
                          ├── ConstructionService → Event → Memory → Episode → Trajectory
                          └── RecallService → ContextPack → prompt_text
```

## 可调用函数

调用顺序与 MemOS 的 `memory_search → memory_get / memory_timeline` 对齐：search 只给摘要和 id，不够再展开。

| 函数 | 作用 | 对应 sidecar | 何时调用 |
|---|---|---|---|
| `cogmem_search` | 返回 excerpt + id，不是全文 | `POST /v1/openclaw/recall` | 自动召回为空、问题太长，或需要换检索角度 |
| `cogmem_get` | 按 id 取一条记忆全文 | `POST /v1/openclaw/get` | search 命中相关，但 excerpt 不够 |
| `cogmem_timeline` | 取一个 episode 内按序的 memory | `POST /v1/openclaw/timeline` | search 命中带了 `episode_id`，需要前后文 |
| `cogmem_status` | 当前会话 / 全库规模 | `GET /v1/status` | search 一直为空，检查是否已 build |
| `cogmem_build` | 手动触发长期记忆构建 | `POST /v1/build` | 长会话后 auto-build 还没跑 |

自动召回没命中时，适配层会注入一段提示，让模型自己生成短 query 再调 `cogmem_search`。search 的 `sufficient=false` 时会带 `next_tools`，提示接着调 get / timeline。
工具说明见 `skill/cogmem-memory-guide/SKILL.md`。

## 依赖

1. 已从源码 `pip install -e ./cogmem-core`，sidecar 可启动：`cogmem sidecar --host 127.0.0.1 --port 8787`
2. 或直接：`python -m cogmem_core.sidecar --store ./cogmem_store --host 0.0.0.0 --port 8787`
3. OpenClaw Gateway 已安装（Node 22+），运行在 WSL 或本机
4. 插件可用 wheel 一键拷贝：`cogmem openclaw-install`

## 配置项

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `sidecarUrl` | `http://127.0.0.1:8787` | sidecar 地址，指向跑 cogmem-core 的服务器 |
| `recallConversationId` | `""` | 留空=用当前会话 id 召回；填 id=写死召回指定会话 |
| `autoRecall` | `true` | before_prompt_build 时自动召回 |
| `autoCapture` | `true` | agent_end 时自动写入事件 |
| `autoBuild` | `true` | compaction/session_end 时自动构建记忆 |
| `buildEveryTurns` | `10` | 每 N 轮自动 build（0=关闭，只靠 compaction 触发） |
| `searchAllFallback` | `true` | 当前会话命中 < recallMinHits 时，自动跨会话召回 |
| `recallMinHits` | `3` | 触发 search_all fallback 的阈值 |
| `minPromptLength` | `4` | query 短于此长度不召回 |
| `timeoutMs` | `8000` | HTTP 请求超时 |

> 注意：`buildEveryTurns`、`searchAllFallback`、`recallMinHits` 是插件内置默认值，
> 不需要（也不能）写进 openclaw.json 的 config 里——OpenClaw 的配置 schema 不允许额外属性。
> 如需修改，改 `index.js` 里的 `DEFAULTS` 对象。

## 安装

### 1. 启动 sidecar

先从源码安装核心：`pip install -e ../cogmem-core`（在仓库根目录执行 `pip install -e ./cogmem-core` 也可以）。

```bash
cogmem sidecar --store ./cogmem_store --host 127.0.0.1 --port 8787
```

验证：

```bash
curl http://127.0.0.1:8787/health
# {"ok":true,"store":"..."}
```

### 2. 安装插件（OpenClaw 所在机器上）

推荐：

```bash
cogmem openclaw-install
```

或用本目录源码：

```bash
openclaw plugins install -l /path/to/openclaw-cogmem
openclaw plugins enable cogmem
```

CLI 若不支持 `-l`，把本目录拷到 `~/.openclaw/plugins/`。

### 3. 配置 openclaw.json

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
          "autoBuild": true,
          "recallConversationId": ""
        },
        "hooks": {
          "allowPromptInjection": true,
          "allowConversationAccess": true
        }
      }
    },
    "load": {
      "paths": ["~/.openclaw/plugins/openclaw-cogmem"]
    }
  }
}
```

> `allowPromptInjection` 必须为 true，否则 before_prompt_build 无法注入 prompt。

### 4. 重启 Gateway

```bash
openclaw gateway start    # 或 restart
```

验证日志里有：
```
cogmem plugin registered sidecar=http://...
```

### 5. 导入历史日志（首次部署）

新 store 是空的，需要把历史会话日志 ingest + build：

```bash
# ingest
curl -X POST http://127.0.0.1:8787/v1/openclaw/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"cogmem-core/tests/fixtures/openclaw/mini_session.jsonl"}'

# build（conversation_id 换成 ingest 返回或 status 列出的会话 id）
curl -X POST http://127.0.0.1:8787/v1/build \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"<conversation_id>"}'

# 确认
curl "http://127.0.0.1:8787/v1/status?conversation_id=<conversation_id>"
```

## 记忆闭环

### 新会话的记忆流程

```
第 1 轮：
  agent_end → ingest 写入当前会话 temporal
  before_prompt_build → recall 当前会话（temporal 有内容）
    → 命中 ≥ recallMinHits(3)：用当前会话记忆
    → 命中 < 3：fallback search_all → 跨会话召回旧记忆

第 2-9 轮：
  agent_end → 继续追加 temporal
  recall → 当前会话 temporal 越来越多 + fallback 旧记忆

第 10 轮（buildEveryTurns）：
  agent_end → ingest + 自动 build → temporal 转长期记忆
  之后 recall → 当前会话的 memory/episode/trajectory

compaction / session_end：
  build → temporal 转长期记忆 → 清空 temporal
```

## 验收

1. sidecar 终端看到 `/v1/openclaw/recall` 和 `/v1/openclaw/ingest` 请求，全部 200
2. Gateway 日志有 `cogmem plugin registered sidecar=...`
3. OpenClaw 里问「我之前问过什么天气」，模型回复提到深圳天气
4. 新会话里说「我叫张三」，下一轮问「我叫什么」，能召回
5. 查 token 成本：见下方"Token 成本分析"

## 排查

### 422 Unprocessable Entity

原因：插件发给 sidecar 的 body 缺 `query` 字段。
确认 sidecar 跑的是最新版 app.py（RecallRequestBody 有默认值）。
插件已修复：extractQuery 只调一次，不会发缺字段的请求。

### recall 返回全 none

原因：该会话没有 ingest 过，或没有 build 过。
```bash
curl "http://127.0.0.1:8787/v1/status?conversation_id=<会话id>"
```
如果 raw_events=0，需要先 ingest。如果 memory_objects=0，需要先 build。

### OpenClaw 显示断开连接

Gateway 进程可能挂了。重启：
```bash
openclaw gateway start
```
或从 OpenClaw Tray 退出再重新打开。

### openclaw.json 配置报错 "additional properties"

OpenClaw 的插件配置 schema 不允许自定义属性。只保留 `sidecarUrl`、`recallConversationId`、`autoRecall`、`autoCapture`、`autoBuild` 这五个。`buildEveryTurns`、`searchAllFallback`、`recallMinHits` 不能写进 config，改 `index.js` 里的 DEFAULTS。

### sidecar 端口被占

```bash
lsof -ti:8787 | xargs kill -9
# 或
fuser -k 8787/tcp
```

### 日志太吵

默认正常请求不打 body。需要调试时：
```bash
COGMEM_LOG_BODY=1 python -m cogmem_core.sidecar --store ./cogmem_store_openclaw --host 0.0.0.0 --port 8787
```

## Token 成本分析

用 tiktoken (cl100k_base) 精确计算，单次召回注入 prompt 的 token 成本：

| 场景 | Token 数 | 占 128k 窗口 |
|---|---|---|
| 固定模板（无命中） | 163 | 0.13% |
| 高召回（平均） | ~1300 | ~1.0% |
| 最高召回 | 1766 | 1.38% |

详见 `cogmem-core/` 下的 token benchmark 脚本。

## 不做的事

- 不在适配层里做 Episode/Trajectory 构建（留给 sidecar/core）
- 不替换 OpenClaw 自带的 MEMORY.md（共存）
- 不把 MemOS 的 Skill / Team Hub 能力搬过来；CogMem 只暴露自己的记忆视图
