# Prompt Parity Record

验证日期：2026-09-22
版本：`brise-rca-score 0.2.0`

验证方式：使用 `brise rca semantic-config get <key>` 回读 active 版本；静态 prompt 从当前源码常量或模板 section 提取；与独立包实际加载内容逐字节比较。

## 结论

v0.2.0 已对齐定因 live 路径中最关键的 semantic mapping 和 mechanism gate prompt，并固化版本/SHA。当前仍不能宣称所有 Brise live prompt 与运行上下文完全相同，主要剩余差异是 process runtime 附加上下文和 Brise PG verdict cache。

| Prompt / 路径 | 独立评分器 | Brise 真源 | 结果 |
|---|---|---|---|
| Semantic mapping | `RCA_SEMANTIC_MAP_PROMPT.md` | active `semantic_map_prompt` 2026-08-04.1 | 逐字一致；v0.2 已接线 |
| Causal rescue | `RCA_LLM_CAUSAL_JUDGE_PROMPT.md` | active `causal_rescue_prompt` 2026-08-04.1 | 逐字一致 |
| Mechanism gate | `RCA_MECHANISM_GATE_PROMPT.md` | active `mechanism_gate_prompt` 2026-08-13.2 | 逐字一致；v0.2 已接线 |
| Process review base | `RCA_PROCESS_REVIEW_PROMPT.md` | `RCA_PROCESS_REVIEW_PROMPT` section | 基础规则相同；独立版追加 direct JSON schema |

## Prompt SHA-256

- Semantic mapping：`56df534f335afa25091f6b775f09a1148fbb2f04dd6dc6c0be12b7f3886e8278`
- Causal rescue：`a97cc34bb5be520ed934c334c4cb68cc34194691408a75009899e395bea71975`
- Mechanism gate：`f42e4ccf8e8dbf89fe3e422414d6995042810b499c2f14c22e0733d72095a167`

## v0.2 Semantic Mapping 行为

- 使用 active prompt 快照，不再使用自定义英文 multi-root fallback。
- single-root 与 multi-root 共用同一 mapper。
- 默认三票，多数票至少 2 票才采纳。
- 空 fault_id 多数票为真实 miss；未知 fault_id/非法 JSON 不作为真实 miss 票。
- 输出 evidence 包含 `semantic_mapping_status`、`semantic_mapped_fault_id`、`semantic_mapping_votes`、prompt version/SHA。

## 仍然存在的运行差异

1. Brise verdict 使用 PG 持久化缓存；独立包只保存当次输出 evidence，不连接 PG。
2. 历史 Brise 判决与新的 live 多票可能因模型漂移不同，即使 prompt 字节一致。
3. Brise process runtime 可追加原始 tool-call 状态、最终结论、高亮 evidence；独立输入合同没有完整 trace tool-call 数据。
4. 独立 fault scorer 尚未完整移植当前 Brise `fault_adjudicator` 的所有扩展通道。
