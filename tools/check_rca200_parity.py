"""Compare locally rescored RCA200 outputs with saved Brise score artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

DIMENSIONS = ("fault_type", "entity", "process", "overall")


def _input_sha(case_dir: Path) -> str:
    gt = (case_dir / "private-gt-bundle.json").read_bytes()
    diagnosis = (case_dir / "diagnosis.json").read_bytes()
    return hashlib.sha256(gt + b"\0" + diagnosis).hexdigest()


def compare_case(old_file: Path, new_file: Path) -> dict[str, object]:
    old_scores = json.loads(old_file.read_text(encoding="utf-8"))["scores"]
    new_payload = json.loads(new_file.read_text(encoding="utf-8"))
    new_scores = new_payload["scores"]
    exact_scores = all(
        Decimal(str(new_scores[name])) == Decimal(str(old_scores[name]))
        for name in DIMENSIONS
    )
    input_match = new_payload.get("input_sha256") == _input_sha(old_file.parent)
    deltas = {
        name: str(Decimal(str(new_scores[name])) - Decimal(str(old_scores[name])))
        for name in DIMENSIONS
    }
    return {
        "case_id": old_file.parent.name,
        "exact": exact_scores and input_match,
        "input_match": input_match,
        "old": {name: old_scores[name] for name in DIMENSIONS},
        "new": {name: new_scores[name] for name in DIMENSIONS},
        "delta": deltas,
    }


def _case_ids(root: Path, filename: str) -> set[str]:
    return {path.parent.name for path in root.glob(f"c*/{filename}")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--rescored-root", type=Path, required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = {item.strip() for item in args.cases.split(",") if item.strip()}
    baseline_ids = _case_ids(args.baseline_root, "brise-rca-score.json")
    rescored_ids = _case_ids(args.rescored_root, "rca-score.json")
    expected_ids = selected or baseline_ids
    missing_baseline = sorted(expected_ids - baseline_ids)
    missing_rescored = sorted(expected_ids - rescored_ids)
    unexpected = sorted(rescored_ids - expected_ids)
    rows = [
        compare_case(
            args.baseline_root / case_id / "brise-rca-score.json",
            args.rescored_root / case_id / "rca-score.json",
        )
        for case_id in sorted(expected_ids & baseline_ids & rescored_ids)
    ]
    expected_count_ok = (
        args.expected_count <= 0 or len(expected_ids) == args.expected_count
    )
    payload = {
        "expected": len(expected_ids),
        "compared": len(rows),
        "exact": sum(1 for row in rows if row["exact"]),
        "different": sum(1 for row in rows if not row["exact"]),
        "missing_baseline": missing_baseline,
        "missing_rescored": missing_rescored,
        "unexpected": unexpected,
        "expected_count_ok": expected_count_ok,
        "cases": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(text, end="")
    valid = bool(rows) and not any(
        (
            payload["different"],
            missing_baseline,
            missing_rescored,
            unexpected,
            not expected_count_ok,
        )
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
