# OpenClaw 集成审查记录

审查日期：2026-09-29。对象：当前 0.3.0 工程版本。

本记录用于保存首次开源时的已知问题，不代表问题已修复。总体架构（Python 核心、HTTP sidecar、OpenClaw 薄适配层）可保留，后续优先完善写入可靠性、构建并发和记忆访问范围。

## 待处理问题

| 编号 | 优先级 | 问题与影响 | 建议 |
| --- | --- | --- | --- |
| OC-001 | P1 | `agent_end` 提交完整消息列表，没有增量筛选；普通消息转换丢弃原始 ID，Python 端生成随机 ID，存储层直接追加。重叠历史会重复入库。 | 保留稳定事件 ID，在服务端实现按会话幂等写入，并处理回调重试。 |
| OC-002 | P1 | 构建读取 temporal 后，完成时清空整个文件；期间写入的新事件可能未参与构建却被清除。原始日志仍保留，但正常增量构建会遗漏这些事件。JSON 写入也缺少原子替换。 | 按会话协调写入和构建，使用检查点或快照清理已处理事件，原子提交构建结果。 |
| OC-003 | P1（共享存储） | 自动跨会话回退扫描整个 store，没有用户、agent 或工作区的访问范围限制。多用户共用时可能串入其他用户记忆。 | 定义记忆归属与授权范围；跨会话检索限定在同一授权域。单人独占存储可暂作为明确的部署前提。 |
| OC-004 | P2 | 实时转换仅保留 role、content、timestamp，丢弃 toolName、toolCallId、isError 等，失败工具结果可能被记为 unknown_tool / success。 | 保留工具执行元数据，并覆盖实时 payload 与 JSONL 导入的一致性。 |
| OC-005 | P2 | 周期构建位于 autoCapture 流程，autoBuild=false 仍触发；计数器在插件实例内共享，多会话互相影响。写入失败后也推进计数。 | 按会话维护计数，写入成功后推进，明确并统一自动构建开关。 |
| OC-006 | P2 | 助手消息入库截断到 800 字符；cogmem_get 的模型可见文本可能进一步截为 400 字符，与完整原文的工具描述不符。 | 保存原文或可解析的原文引用，提供全文或分页读取；区分摘要与原文。 |
| OC-007 | P2 | search_all 只选择一个会话的结果，主要按候选数量衡量，没有跨会话合并与统一相关性排序。 | 在授权范围内合并候选、去重并排序，保留来源会话标识。 |

## 代码定位

- OC-001、OC-004、OC-005：[插件入口](../openclaw-cogmem/index.js)。
- OC-001、OC-002：[本地存储](../cogmem-core/src/cogmem_core/storage.py)、[构建服务](../cogmem-core/src/cogmem_core/construction/service.py)。
- OC-003、OC-007：[HTTP 服务](../cogmem-core/src/cogmem_core/sidecar/app.py)。
- OC-006：[Python 适配器](../cogmem-core/src/cogmem_core/adapters/openclaw.py)、[工具返回格式](../openclaw-cogmem/tools.js)。

## 验证范围

已进行静态代码检查和 Node.js 模拟 hook 验证：

- 连续提交含重叠历史的消息列表，插件分别发送 3 条、4 条消息，未过滤重叠部分。
- 带 toolName、toolCallId、isError 的工具结果，转换后仅剩 role、content。
- autoBuild=false 时，A 会话累计 9 次、B 会话 1 次，仍对 B 触发周期构建。

上述模拟验证未连接真实 Gateway；未执行真实 OpenClaw 端到端测试，也未执行 Python 并发复现。并发与跨域问题依据代码路径分析，需补充回归测试。

OpenClaw hook 语义参考：[Agent loop](https://docs.openclaw.ai/agent-loop)、[Plugin hooks](https://docs.openclaw.ai/plugins/hooks)。实际兼容性需以固定 OpenClaw 版本验证。

## 后续验收建议

1. 重放同一批事件不增加原始记录或长期记忆数量。
2. 构建期间追加事件不会遗漏；同会话并行构建不会覆盖新结果。
3. 不同用户或工作区的记忆不会越过授权边界。
4. 工具成功 / 失败状态和关联 ID 在转换后保持一致。
5. 自动构建关闭后不触发构建，多会话计数相互独立。
6. 全文工具能够读回超过摘要长度的原始内容。
7. 两个授权会话中的相关证据可以同时进入跨会话检索结果。
