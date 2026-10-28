# Changelog

## Unreleased - 2026-10-22

- Split the monolithic `offline.py` into contracts, input loading, public API,
  resolution, process runtime, single-root, and multi-root orchestration modules.
- Preserve the `brise_rca_score.offline` compatibility facade and all v0.3.2
  score formulas, prompts, gates, model settings, and serialized result contracts.
- Add the repository examples referenced by the README and document module
  ownership in `docs/ARCHITECTURE.md`.
- Make a fresh development install include PyYAML type stubs so `mypy src` is
  reproducible.
- Remove duplicate dependency metadata, one-time release-preparation artifacts,
  and unreachable legacy extraction / credibility modules.
- Move maintained validation and parity records under `docs/`.
- Replace the repository README with a formal usage, architecture, input/output,
  scoring-rule, development, and security reference.
- Standardize the generic local topology adapter as `JsonTopology` and remove
  stale dataset-specific naming.

## 0.3.2 - 2026-10-12

- Preserve the v0.3.1 F/E/P/O scoring semantics.
- Parse independently decodable JSON objects from reasoning-model responses while retaining strict schema and relation validation.
- Retry invalid structured judge responses with an explicit JSON-only correction instruction.
- Keep Kimi K3 thinking enabled and increase only its exhausted fallback completion budget.
- Add model-protocol regression coverage; the full suite now contains 156 tests.

## 0.3.1 - 2026-09-22

- Freeze Brise semantic mapping and mechanism-gate prompt snapshots.
- Add three-vote semantic mapping, RCA200 topology normalization and optional trajectory context.
- Verify decision replay parity and RCA200 live scoring behavior documented in `docs/VALIDATION.md`.
