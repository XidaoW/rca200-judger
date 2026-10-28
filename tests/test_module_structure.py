"""Regression checks for the maintainable scorer module layout and examples."""

import json
from pathlib import Path

from brise_rca_score import api, contracts, offline
from brise_rca_score.inputs import diagnosis_from_agent_output, load_case_inputs
from brise_rca_score.llm.stub import StubLLM


def test_offline_facade_preserves_legacy_public_imports() -> None:
    assert offline.score_case is api.score_case
    assert offline.diagnosis_from_agent_output is diagnosis_from_agent_output
    assert offline.load_case_inputs is load_case_inputs
    assert offline.Diagnosis is contracts.Diagnosis
    assert offline.GroundTruthBundle is contracts.GroundTruthBundle
    assert offline.OfflineScore is contracts.OfflineScore


def test_documented_decision_replay_example_is_current() -> None:
    root = Path(__file__).parents[1]
    result = api.score_case(
        gt_file=root / "examples/single-root/private-gt-bundle.json",
        agent_output_file=root / "examples/single-root/agent_output.json",
        replay_decision_file=root / "examples/single-root/decision-snapshot.json",
        llm=StubLLM([]),
    )
    expected = json.loads(
        (root / "examples/single-root/expected-decision-replay.json").read_text()
    )
    # Source paths depend on whether callers use absolute or relative paths.
    expected["source"] = result.source
    assert result.model_dump(mode="json") == expected


def test_documented_multi_root_example_matches_contracts() -> None:
    root = Path(__file__).parents[1]
    gt = contracts.GroundTruthBundle.model_validate_json(
        (root / "examples/multi-root/private-gt-bundle.json").read_bytes()
    )
    diagnosis = diagnosis_from_agent_output(
        root / "examples/multi-root/agent_output.json"
    )
    assert len(gt.ground_truth.outcome.root_causes) == 2
    assert len(diagnosis.atoms) == 2
