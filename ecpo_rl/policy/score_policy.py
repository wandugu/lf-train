# -*- coding: utf-8 -*-
"""Score trajectories and emit ECPO ranking/certificate outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ecpo_rl.ecpo import (  # type: ignore
        DeterministicEvidenceVerifier,
        ECPOValidator,
        build_policy_output,
        build_window_from_trajectories,
        evaluate_certified_output,
    )
    from ecpo_rl.irl.maxent_irl import MaxEntIRL  # type: ignore
else:
    from ..ecpo import (
        DeterministicEvidenceVerifier,
        ECPOValidator,
        build_policy_output,
        build_window_from_trajectories,
        evaluate_certified_output,
    )
    from ..irl.maxent_irl import MaxEntIRL


def load_trajectories(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    if not items:
        raise ValueError("trajectory file is empty")
    return items


def load_policy_logprobs(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text("utf-8"))
    return {k: float(v) for k, v in payload.items()}


def heuristic_policy_logprob(traj: Dict) -> float:
    hits = sum(len(step.get("skeleton_hits", [])) for step in traj.get("steps", []))
    penalties = sum(abs(step.get("delta_days_from_prev", 0)) for step in traj.get("steps", [])[1:])
    return float(hits - 0.1 * penalties)


def normalise(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return []
    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        return [0.0 for _ in values]
    return ((arr - mean) / std).tolist()


def summarise_reason(traj: Dict, reward: float, logprob: float) -> Dict:
    top_steps = sorted(traj.get("steps", []), key=lambda s: len(s.get("skeleton_hits", [])), reverse=True)
    highlights = []
    for step in top_steps[:2]:
        refs = ",".join(f"{ref['doc_id']}:{ref['span']}" for ref in step.get("text_refs", []))
        highlights.append({
            "event_id": step.get("event_id"),
            "skeleton_hits": step.get("skeleton_hits", []),
            "refs": refs,
        })
    return {
        "trajectory_id": traj.get("trajectory_id"),
        "person_id": traj.get("person_id"),
        "reward": reward,
        "policy_logprob": logprob,
        "highlights": highlights,
    }


def evaluate_ranking(ranked: List[Tuple[str, float]], trajectories: List[Dict], k: int) -> Dict[str, float]:
    relevance_map = {traj["trajectory_id"]: {"expert": 2, "candidate": 1, "negative": 0}[traj.get("label", "candidate")] for traj in trajectories}
    hits = 0
    dcg = 0.0
    idcg = 0.0
    sorted_rels = sorted(relevance_map.values(), reverse=True)
    for idx, (traj_id, _) in enumerate(ranked[:k], start=1):
        rel = relevance_map.get(traj_id, 0)
        hits += 1 if rel > 0 else 0
        dcg += (2 ** rel - 1) / np.log2(idx + 1)
    for idx, rel in enumerate(sorted_rels[:k], start=1):
        idcg += (2 ** rel - 1) / np.log2(idx + 1)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MAP approximation
    relevant = 0
    ap = 0.0
    for idx, (traj_id, _) in enumerate(ranked[:k], start=1):
        rel = relevance_map.get(traj_id, 0)
        if rel > 0:
            relevant += 1
            ap += relevant / idx
    map_score = ap / relevant if relevant else 0.0
    hit_rate = hits / k if k else 0.0
    return {"NDCG@K": ndcg, "MAP@K": map_score, "Hit@K": hit_rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank trajectories and emit ECPO certificates")
    parser.add_argument("--traj-path", type=Path, default=Path("data/processed/traj.jsonl"))
    parser.add_argument("--reward-ckpt", type=Path, default=Path("outputs/qwen-4b-rm/reward.ckpt"))
    parser.add_argument("--policy-logprobs", type=Path, default=Path("outputs/qwen-4b-rl/policy_logprobs.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-4b-rl"))
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    trajectories = load_trajectories(args.traj_path)

    model = MaxEntIRL()
    model.load(args.reward_ckpt)

    reward_values = [model.score(traj) for traj in trajectories]
    logprob_map = load_policy_logprobs(args.policy_logprobs)
    logprob_values = []
    for traj in trajectories:
        traj_id = traj.get("trajectory_id")
        logprob_values.append(logprob_map.get(traj_id, heuristic_policy_logprob(traj)))

    norm_rewards = normalise(reward_values)
    norm_logprobs = normalise(logprob_values)

    combined_scores = []
    reasons = []
    for traj, r, l in zip(trajectories, norm_rewards, norm_logprobs):
        score = args.alpha * r + (1 - args.alpha) * l
        combined_scores.append((traj.get("trajectory_id"), score))
        reasons.append(summarise_reason(traj, r, l))

    combined_scores.sort(key=lambda x: x[1], reverse=True)
    topk = combined_scores[: args.k]
    window = build_window_from_trajectories(
        trajectories,
        window_id="score_window",
        intent_id="ECPO",
    )
    ecpo_output = build_policy_output(window, [traj_id for traj_id, _score in combined_scores], args.k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "topk.json").write_text(json.dumps([{"trajectory_id": traj_id, "score": score} for traj_id, score in topk], ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "ecpo_output.json").write_text(json.dumps(ecpo_output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "reasons.jsonl").open("w", encoding="utf-8") as f:
        for reason in reasons:
            f.write(json.dumps(reason, ensure_ascii=False) + "\n")

    print(f"Top-{args.k} trajectories written to {args.output_dir / 'topk.json'}")
    print(f"ECPO ranking/certificate output written to {args.output_dir / 'ecpo_output.json'}")

    if args.eval:
        metrics = evaluate_ranking(combined_scores, trajectories, args.k)
        certified = evaluate_certified_output(
            ecpo_output,
            window,
            ECPOValidator(k=args.k),
            DeterministicEvidenceVerifier(),
            args.k,
        )
        metrics.update(certified)
        print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
