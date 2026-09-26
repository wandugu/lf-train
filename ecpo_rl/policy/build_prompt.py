# -*- coding: utf-8 -*-
"""Prompt builders for ECPO trajectory and window serialisation."""

from __future__ import annotations

from typing import Dict, Iterable, List


def format_step(step: Dict) -> str:
    roles = step.get("roles", {})
    role_str = ", ".join(f"{k}:{v}" for k, v in roles.items())
    skeleton = ",".join(step.get("skeleton_hits", []))
    refs = "; ".join(f"{ref.get('doc_id')}:{ref.get('span')}" for ref in step.get("text_refs", []))
    return f"{step.get('time')} | {step.get('type')} | 角色={role_str} | 骨架={skeleton} | 证据={refs}"


def build_prompt(trajectory: Dict, include_future_hint: bool = False) -> str:
    header = [f"[人员] {trajectory.get('person_id')}", f"[轨迹] {trajectory.get('trajectory_id')}"]
    lines = ["[历史事件]"]
    for step in trajectory.get("steps", []):
        lines.append(format_step(step))
    prompt = "\n".join(header + lines)
    if include_future_hint and trajectory.get("steps"):
        prompt += "\n请总结该人员下一步可能的骨架动作并给出理由。"
    else:
        prompt += "\n请总结该人员已知轨迹并评估风险。"
    return prompt


def build_batch_prompts(trajectories: Iterable[Dict], include_future_hint: bool = False) -> List[str]:
    return [build_prompt(traj, include_future_hint=include_future_hint) for traj in trajectories]


if __name__ == "__main__":
    demo = {
        "person_id": "P_001",
        "trajectory_id": "traj_P001",
        "steps": [
            {
                "time": "2014-03-21",
                "type": "Conflict.Explosion",
                "roles": {"Agent": "P_001", "Target": "O_017"},
                "skeleton_hits": ["EXECUTE"],
                "text_refs": [{"doc_id": "doc_demo", "span": [10, 20]}],
            }
        ],
    }
    print(build_prompt(demo, include_future_hint=True))
