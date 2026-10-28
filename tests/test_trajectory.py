from __future__ import annotations

import json
from pathlib import Path

from brise_rca_score.trajectory import load_process_trajectory


def test_load_process_trajectory_builds_status_summary(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "agent", "message": "investigating"},
                    {
                        "source": "agent",
                        "message": "(tool use)",
                        "tool_calls": [
                            {
                                "tool_call_id": "1",
                                "function_name": "Bash",
                                "arguments": {"command": "echo ok"},
                            }
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "1",
                                    "content": "ok",
                                    "is_error": False,
                                }
                            ]
                        },
                    },
                    {"source": "agent", "message": "final root cause conclusion"},
                ]
            }
        )
    )
    ctx = load_process_trajectory(path)
    assert ctx is not None
    assert ctx.analysis_output == "final root cause conclusion"
    assert "Bash" in ctx.tool_calls_summary
    assert "(成功)" in ctx.tool_calls_summary
