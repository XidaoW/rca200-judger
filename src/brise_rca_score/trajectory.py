"""Deterministic local extraction of process-review context from ATIF trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ProcessTrajectoryContext(BaseModel):
    analysis_output: str = ""
    tool_calls_summary: str = ""


def _compact(value: object, limit: int = 900) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return " ".join(text.split())[:limit]


def load_process_trajectory(path: Path | None) -> ProcessTrajectoryContext | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") or []
    agent_messages: list[str] = []
    lines: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        message = str(step.get("message") or "").strip()
        calls = step.get("tool_calls") or []
        if (
            step.get("source") == "agent"
            and message
            and message != "(tool use)"
            and not calls
        ):
            agent_messages.append(message)
        observations = (step.get("observation") or {}).get("results") or []
        by_id = {
            str(x.get("source_call_id") or ""): x
            for x in observations
            if isinstance(x, dict)
        }
        for call in calls:
            if not isinstance(call, dict):
                continue
            result = by_id.get(str(call.get("tool_call_id") or ""), {})
            content = result.get("content")
            if result.get("is_error") is True:
                status = "失败"
            elif content in (None, "", [], {}):
                status = "空结果"
            else:
                status = "成功"
            name = "_".join(str(call.get("function_name") or "unknown").split())[:64]
            query = _compact(call.get("arguments") or {}, 700)
            returned = (
                "无返回"
                if content in (None, "", [], {})
                else f"返回 {len(str(content))} 字符"
            )
            lines.append(f"- {name}: {query} → {returned} ({status})")
    analysis = agent_messages[-1] if agent_messages else ""
    summary = "\n".join(lines)
    if len(summary) > 4000:
        summary = summary[:3900] + f"\n（明细截断；总调用 {len(lines)} 次）"
    return ProcessTrajectoryContext(
        analysis_output=analysis, tool_calls_summary=summary
    )
