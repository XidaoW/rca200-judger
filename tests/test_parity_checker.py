from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL_PATH = Path(__file__).parents[1] / "tools" / "check_rca200_parity.py"
SPEC = importlib.util.spec_from_file_location("check_rca200_parity", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_case(root: Path, case_id: str, *, score: float) -> None:
    case = root / case_id
    case.mkdir(parents=True)
    (case / "private-gt-bundle.json").write_text('{"ground_truth":{}}')
    (case / "diagnosis.json").write_text("{}")
    payload = {
        "scores": {
            "fault_type": score,
            "entity": score,
            "process": score,
            "overall": score,
        }
    }
    (case / "brise-rca-score.json").write_text(json.dumps(payload))


def _write_rescore(root: Path, baseline: Path, case_id: str, *, score: float) -> None:
    case = root / case_id
    case.mkdir(parents=True)
    digest = MODULE._input_sha(baseline / case_id)
    payload = {
        "input_sha256": digest,
        "scores": {
            "fault_type": score,
            "entity": score,
            "process": score,
            "overall": score,
        },
    }
    (case / "rca-score.json").write_text(json.dumps(payload))


def test_parity_rejects_unknown_selected_case(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline"
    rescored = tmp_path / "rescored"
    baseline.mkdir()
    rescored.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--baseline-root",
            str(baseline),
            "--rescored-root",
            str(rescored),
            "--cases",
            "c999",
        ],
    )
    assert MODULE.main() == 1


def test_parity_requires_matching_input_digest(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline"
    rescored = tmp_path / "rescored"
    _write_case(baseline, "c001", score=1.0)
    _write_rescore(rescored, baseline, "c001", score=1.0)
    payload_path = rescored / "c001" / "rca-score.json"
    payload = json.loads(payload_path.read_text())
    payload["input_sha256"] = "stale"
    payload_path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--baseline-root",
            str(baseline),
            "--rescored-root",
            str(rescored),
            "--expected-count",
            "1",
        ],
    )
    assert MODULE.main() == 1


def test_parity_accepts_exact_scores_and_digest(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline"
    rescored = tmp_path / "rescored"
    _write_case(baseline, "c001", score=1.0)
    _write_rescore(rescored, baseline, "c001", score=1.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--baseline-root",
            str(baseline),
            "--rescored-root",
            str(rescored),
            "--expected-count",
            "1",
        ],
    )
    assert MODULE.main() == 0
