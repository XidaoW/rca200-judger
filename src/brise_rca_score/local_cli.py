"""CLI for SLS-free scoring of existing RCA200 artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

from .llm.base import LLMClient
from .llm.dashscope import DashScopeLLM
from .llm.stub import StubLLM
from .offline import OfflineScore, score_case


def _build_llm(args: argparse.Namespace) -> LLMClient:
    if getattr(args, "replay_decision_file", "") or getattr(
        args, "replay_decisions", False
    ):
        return StubLLM("{}")
    return DashScopeLLM(default_model=args.model, timeout_s=args.timeout)


def _write_result(result: OfflineScore, output: Path | None) -> None:
    payload = result.model_dump_json(indent=2)
    if output is None:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(output)


def _score_one(args: argparse.Namespace) -> int:
    llm = _build_llm(args)
    result = score_case(
        gt_file=Path(args.gt_file),
        diagnosis_file=Path(args.diagnosis_file) if args.diagnosis_file else None,
        agent_output_file=Path(args.agent_output) if args.agent_output else None,
        llm=llm,
        model=args.model,
        temperature=args.temperature,
        credibility=args.credibility,
        replay_process_file=(
            Path(args.replay_process_file) if args.replay_process_file else None
        ),
        replay_decision_file=(
            Path(args.replay_decision_file) if args.replay_decision_file else None
        ),
        topology_file=Path(args.topology_file) if args.topology_file else None,
        trajectory_file=Path(args.trajectory_file) if args.trajectory_file else None,
        vote_count=args.vote_count,
    )
    _write_result(result, Path(args.output) if args.output else None)
    return 0


def _parse_case_filter(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _find_topology(dataset_root: Path | None, case_id: str) -> Path | None:
    if dataset_root is None:
        return None
    candidates = [
        dataset_root / "cases" / case_id / "topology.json",
        dataset_root / case_id / "topology.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _score_case_directory(
    case_dir: Path,
    *,
    output_root: Path,
    dataset_root: Path | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    case_id = case_dir.name
    replay = case_dir / "brise-rca-score.json" if args.reuse_process_review else None
    decisions = case_dir / "brise-rca-score.json" if args.replay_decisions else None
    for snapshot in (replay, decisions):
        if snapshot is not None and not snapshot.is_file():
            raise ValueError(f"missing replay score: {snapshot}")
    diagnosis_file = case_dir / "diagnosis.json"
    agent_output_file = case_dir / "agent_output.json"
    use_diagnosis = decisions is not None or not agent_output_file.is_file()
    result = score_case(
        gt_file=case_dir / "private-gt-bundle.json",
        diagnosis_file=diagnosis_file if use_diagnosis else None,
        agent_output_file=agent_output_file if not use_diagnosis else None,
        llm=_build_llm(args),
        model=args.model,
        temperature=args.temperature,
        credibility=args.credibility,
        replay_process_file=replay,
        replay_decision_file=decisions,
        topology_file=_find_topology(dataset_root, case_id),
        trajectory_file=(
            case_dir / "trajectory.json"
            if args.process_context == "trajectory"
            and (case_dir / "trajectory.json").is_file()
            else None
        ),
        vote_count=args.vote_count,
    )
    _write_result(result, output_root / case_id / "rca-score.json")
    return {"case_id": case_id, "status": "ok", "scores": result.scores}


def _score_batch(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    output_root = Path(args.output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory is not empty: {output_root}")
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    selected = _parse_case_filter(args.cases)
    case_dirs = [
        path
        for path in sorted(run_root.glob("c*"))
        if path.is_dir() and (selected is None or path.name in selected)
    ]
    if args.limit > 0:
        case_dirs = case_dirs[: args.limit]
    if not case_dirs:
        raise ValueError(f"no case directories found under {run_root}")

    summary: list[dict[str, object]] = []
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_case = {
            executor.submit(
                _score_case_directory,
                case_dir,
                output_root=output_root,
                dataset_root=dataset_root,
                args=args,
            ): case_dir.name
            for case_dir in case_dirs
        }
        for future in concurrent.futures.as_completed(future_to_case):
            case_id = future_to_case[future]
            try:
                summary.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failed += 1
                summary.append(
                    {
                        "case_id": case_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    summary.sort(key=lambda item: str(item["case_id"]))
    summary_file = output_root / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(
        {"total": len(case_dirs), "failed": failed, "cases": summary},
        ensure_ascii=False,
        indent=2,
    )
    temporary = summary_file.with_suffix(".json.tmp")
    temporary.write_text(summary_text + "\n", encoding="utf-8")
    temporary.replace(summary_file)
    print(summary_file)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rca-score",
        description="Score existing RCA artifacts without SLS or Brise server access.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("score", help="score one local case")
    one.add_argument("--gt-file", required=True)
    scored_output = one.add_mutually_exclusive_group(required=True)
    scored_output.add_argument("--diagnosis-file", default="")
    scored_output.add_argument("--agent-output", default="")
    one.add_argument("--topology-file", default="")
    one.add_argument("--trajectory-file", default="")
    replay_mode = one.add_mutually_exclusive_group()
    replay_mode.add_argument("--replay-process-file", default="")
    replay_mode.add_argument("--replay-decision-file", default="")
    one.add_argument("--output", default="")
    _add_model_args(one)
    one.set_defaults(func=_score_one)

    batch = sub.add_parser("batch", help="score case directories below a run root")
    batch.add_argument("--run-root", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--dataset-root", default="")
    batch.add_argument("--cases", default="")
    batch.add_argument("--limit", type=int, default=0)
    batch.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    batch.add_argument(
        "--process-context",
        choices=["legacy", "trajectory"],
        default="legacy",
        help="legacy keeps historical local inputs; trajectory adds ATIF tool-call evidence",
    )
    batch_replay = batch.add_mutually_exclusive_group()
    batch_replay.add_argument("--reuse-process-review", action="store_true")
    batch_replay.add_argument("--replay-decisions", action="store_true")
    _add_model_args(batch)
    batch.set_defaults(func=_score_batch)
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="qwen3.7-plus",
        help="DashScope model used by semantic, mechanism and process judges.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--credibility",
        choices=["RELIABLE", "QUESTIONABLE", "UNRELIABLE"],
        default="RELIABLE",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--vote-count", type=int, default=3)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"rca-score: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
