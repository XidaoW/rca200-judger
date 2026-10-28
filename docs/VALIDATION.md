# Validation Record

基线版本：`brise-rca-score 0.3.1`
基线验证日期：2026-09-22

- Ruff / format / mypy：通过。
- pytest：136 passed。
- RCA200 live legacy：200/200 成功，失败 0。
- F/E/P/O 平均：0.514310 / 0.5655 / 0.645333 / 56.887。
- 对比 Brise MAE：F 0.1288、E 0.1403、P 0.0962、Overall 11.221。
- Pearson：F 0.7764、E 0.7215、P 0.7026、Overall 0.7102。
- Gate 一致：154/200（77%）；Brise 通过 91，v0.3.1 通过 99。
- trajectory context 保留为显式 opt-in，不再默认影响历史可比评分。

## v0.3.2 兼容增量验证

验证日期：2026-10-12

- 评分公式、权重、门禁、语义映射和舍入规则未变。
- 增量范围：结构化 JSON 对象扫描、协议纠正重试、Kimi K3 hidden reasoning 预算兜底。
- pytest：156/156 通过。
- Qwen RCA200 冻结队列首轮评分：50/50 成功；该批结果由 v0.3.1 基线产生。
- DeepSeek、GLM、Kimi K3 的兼容修复结果必须保留 first-pass/retry provenance，不得覆盖首轮产物。
- 本轮没有重新声明 live scoring 与 Brise 完全等价；一致性边界仍以 README 的说明为准。
## 2026-10-22 结构整理验证

- 仅调整模块职责和仓库可维护性，不修改评分公式、权重、prompt、gate、模型参数或输出合同。
- `brise_rca_score.offline` 保留原公共导入兼容层。
- README 引用的 single-root / multi-root 脱敏样例已纳入仓库。
- pytest：123/123 通过。
- 新增/修改模块 Ruff check 与 format：通过。
- `mypy src`：通过。
- wheel 与 sdist 构建：通过。
- 删除仅由自身测试调用的 legacy extraction / credibility 代码后，保留的 123 项测试覆盖全部 CLI、公共 API、单根因、多根因、F/E/P/O kernel、拓扑、LLM 协议和 replay 路径。
