# Scorer Architecture

This document describes the maintenance boundaries of `brise-rca-score` v0.3.2.
The refactor dated 2026-10-22 changes module ownership only; it does not change
F/E/P/O formulas, prompts, model settings, gates, or serialized score contracts.

## Runtime flow

```text
CLI (`local_cli.py`)
  -> public API (`api.py`)
     -> input contracts and conversion (`contracts.py`, `inputs.py`)
     -> single-root orchestration (`single_root.py`)
     -> multi-root orchestration (`multi_root_engine.py`)
     -> process live/replay orchestration (`process_runtime.py`)
     -> shared resolution helpers (`resolution.py`)
     -> pure scoring kernels (`scoring/`)
     -> model adapter (`llm/`) and local topology (`topology/`)
```

`offline.py` is intentionally a compatibility facade. Existing consumers may
continue importing `score_case`, `diagnosis_from_agent_output`, and the Pydantic
contracts from `brise_rca_score.offline`. New internal code should import from
the owning module directly.

## Module ownership

| Module | Owns | Must not own |
|---|---|---|
| `contracts.py` | Pydantic input/output contracts | File I/O, network calls, scoring |
| `inputs.py` | Local file loading, hashing, Agent-output conversion | Score formulas, provider calls |
| `api.py` | Public `score_case` routing | Dimension algorithms |
| `single_root.py` | Single-root F/E/P orchestration and result assembly | CLI parsing, provider transport |
| `multi_root_engine.py` | Multi-root assignment/rescue orchestration | Generic assignment kernel |
| `process_runtime.py` | Process judge retries and decision replay | HTTP transport details |
| `resolution.py` | GT reasoning and fault/entity normalization | Final weighted score calculation |
| `scoring/` | Pure or dimension-local scoring behavior | CLI and filesystem traversal |
| `llm/` | DashScope transport and protocol compatibility | RCA business formulas |
| `topology/json_graph.py` | Local JSON topology loading and graph queries | Remote topology access |
| `local_cli.py` | CLI arguments, batch traversal, result persistence | Scoring semantics |

## Change rules

1. Formula, threshold, weight, prompt, or mapping changes require explicit
   scoring-version and validation notes. Do not hide semantic changes inside a
   structural refactor.
2. Provider-specific compatibility belongs in `llm/`; it must not leak into
   pure scoring kernels.
3. Batch retry must write a new output directory. Existing result artifacts are
   immutable provenance.
4. Runtime code must remain independent of SLS, PostgreSQL, and Brise clients.
5. Preserve `brise_rca_score.offline` imports until a separately announced
   breaking release.

## Verification

For a structural-only change, run:

```bash
python -m pytest -q
ruff check src tests tools
ruff format --check src tests tools
mypy src
python -m build
```

Decision replay remains the strongest semantic regression gate because it
recomputes F/E/P/O from frozen decisions without invoking an LLM.
