from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from brise_rca_score.llm.stub import StubLLM
from brise_rca_score.local_cli import _build_llm, _score_batch, build_parser


def test_score_requires_exactly_one_output_input() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--gt-file", "gt.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "score",
                "--gt-file",
                "gt.json",
                "--diagnosis-file",
                "diagnosis.json",
                "--agent-output",
                "agent.json",
            ]
        )


def test_decision_replay_does_not_require_dashscope(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    args = argparse.Namespace(replay_decision_file="score.json", replay_decisions=False)
    assert isinstance(_build_llm(args), StubLLM)


def test_batch_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.json").write_text("{}")
    args = argparse.Namespace(
        run_root=str(tmp_path / "cases"),
        output_dir=str(output),
        dataset_root="",
        cases="",
        limit=0,
    )
    with pytest.raises(ValueError, match="not empty"):
        _score_batch(args)


def test_batch_process_context_defaults_to_legacy() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["batch", "--run-root", "/tmp/run", "--output-dir", "/tmp/out"]
    )
    assert args.process_context == "legacy"
    args = parser.parse_args(
        [
            "batch",
            "--run-root",
            "/tmp/run",
            "--output-dir",
            "/tmp/out",
            "--process-context",
            "trajectory",
        ]
    )
    assert args.process_context == "trajectory"
