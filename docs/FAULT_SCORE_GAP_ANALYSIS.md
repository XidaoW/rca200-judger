# RCA v0.3.1 全量实验与 Brise 对比

日期：2026-09-22

| 维度 | Brise | v0.2 | v0.3 trajectory | v0.3.1 legacy | v0.3.1 MAE |
|---|---:|---:|---:|---:|---:|
| F | 0.4260 | 0.5205 | 0.5117 | 0.5143 | 0.1288 |
| E | 0.5843 | 0.5188 | 0.5655 | 0.5655 | 0.1403 |
| P | 0.5842 | 0.6481 | 0.6846 | 0.6453 | 0.0962 |
| O | 53.159 | 52.890 | 57.9745 | 56.887 | 11.221 |

## 结论

- 保留 v0.3 topology 修复后，E MAE 从 v0.2 的 0.2230 降至 0.1403。
- 将 trajectory context 改为 opt-in 后，P MAE 从 v0.3 的 0.1297 回落至 0.0962。
- Overall MAE 为 11.221，是 v0.2/v0.3/v0.3.1 三版中最低；Pearson 0.7102，Gate 一致 154/200。
- 默认 `legacy` 适合与历史 Brise 对比；`--process-context trajectory` 用于带真实工具证据的新口径评估。
- 剩余主要差异仍是历史 semantic/rescue/mechanism verdict cache 与当前 live 判决，以及完整 Brise relation/adoption adjudicator 未完全本地化。
