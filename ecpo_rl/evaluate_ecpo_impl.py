# -*- coding: utf-8 -*-
"""Evaluate ECPO ranking and certificate metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ecpo_rl.ecpo import (  # type: ignore
        DeterministicEvidenceVerifier,
        ECPOValidator,
        build_policy_output,
        build_window_from_trajectories,
        evaluate_certified_output,
    )
    from ecpo_rl.config_utils import resolve_mode  # type: ignore
    from ecpo_rl.irl.features import build_feature_vector  # type: ignore
    from ecpo_rl.irl.maxent_irl import MaxEntIRL  # type: ignore
    from ecpo_rl.policy.score_policy import (  # type: ignore
        evaluate_ranking,
        heuristic_policy_logprob,
        load_policy_logprobs,
        load_trajectories,
        normalise,
        summarise_reason,
    )
else:
    from .ecpo import (
        DeterministicEvidenceVerifier,
        ECPOValidator,
        build_policy_output,
        build_window_from_trajectories,
        evaluate_certified_output,
    )
    from .config_utils import resolve_mode
    from .irl.features import build_feature_vector
    from .irl.maxent_irl import MaxEntIRL
    from .policy.score_policy import (
        evaluate_ranking,
        heuristic_policy_logprob,
        load_policy_logprobs,
        load_trajectories,
        normalise,
        summarise_reason,
    )


LOGGER = logging.getLogger(__name__)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config yaml must be a mapping")
    return payload


def _build_sample_trajectory() -> Dict:
    return {
        "person_id": "P_SAMPLE",
        "trajectory_id": "traj_sample_1",
        "label": "expert",
        "steps": [
            {
                "event_id": "event_sample_1",
                "skeleton_hits": ["PREP", "EXECUTE"],
                "delta_days_from_prev": 0,
                "roles": {"Agent": "A_SAMPLE", "Target": "T_SAMPLE"},
                "time": "2014-03-21",
                "text_refs": [{"doc_id": "doc1", "span": "0-12"}],
            }
        ],
        "meta": {"graph_nodes": ["A_SAMPLE", "T_SAMPLE"], "graph_edges": [["A_SAMPLE", "EXECUTE", "T_SAMPLE", "2014-03-21"]]},
    }


def _ensure_sample_assets(root: Path, cfg: Dict) -> Dict[str, Path]:
    sample_dir = _resolve_path(root, cfg["sample_dir"])
    sample_traj = _resolve_path(root, cfg["sample_traj_path"])
    sample_reward = _resolve_path(root, cfg["sample_reward_ckpt"])
    sample_logprobs = _resolve_path(root, cfg["sample_policy_logprobs"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    rewrite_sample_traj = not sample_traj.exists()
    if sample_traj.exists():
        try:
            first = json.loads(sample_traj.read_text(encoding="utf-8").splitlines()[0])
            rewrite_sample_traj = not first.get("steps", [{}])[0].get("event_id")
        except Exception:  # noqa: BLE001
            rewrite_sample_traj = True

    if rewrite_sample_traj:
        traj_payload = _build_sample_trajectory()
        sample_traj.write_text(json.dumps(traj_payload, ensure_ascii=False) + "\n", encoding="utf-8")
        LOGGER.debug("Sample trajectory written: %s", sample_traj)

    if not sample_reward.exists():
        feature_dim = build_feature_vector(_build_sample_trajectory()).shape[0]
        payload = {"theta": [0.0] * feature_dim, "temperature": 1.0}
        sample_reward.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.debug("Sample reward ckpt written: %s", sample_reward)

    if not sample_logprobs.exists():
        sample_logprobs.write_text(json.dumps({"traj_sample_1": 0.0}, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.debug("Sample policy logprobs written: %s", sample_logprobs)

    return {
        "traj_path": sample_traj,
        "reward_ckpt": sample_reward,
        "policy_logprobs": sample_logprobs,
    }


def _score_trajectories(
    trajectories: List[Dict],
    reward_ckpt: Path,
    policy_logprobs: Path,
    alpha: float,
) -> Tuple[List[Tuple[str, float]], List[Dict]]:
    model = MaxEntIRL()
    model.load(reward_ckpt)

    reward_values = [model.score(traj) for traj in trajectories]
    logprob_map = load_policy_logprobs(policy_logprobs)
    logprob_values = []
    for traj in trajectories:
        traj_id = traj.get("trajectory_id")
        logprob_values.append(logprob_map.get(traj_id, heuristic_policy_logprob(traj)))

    norm_rewards = normalise(reward_values)
    norm_logprobs = normalise(logprob_values)

    combined_scores: List[Tuple[str, float]] = []
    reasons: List[Dict] = []
    for traj, r, l in zip(trajectories, norm_rewards, norm_logprobs):
        score = alpha * r + (1 - alpha) * l
        combined_scores.append((traj.get("trajectory_id"), score))
        reasons.append(summarise_reason(traj, r, l))

    combined_scores.sort(key=lambda x: x[1], reverse=True)
    return combined_scores, reasons


def _write_outputs(
    output_dir: Path,
    dataset: str,
    profile: str,
    topk: List[Tuple[str, float]],
    reasons: Iterable[Dict],
    ecpo_output: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_payload = [{"trajectory_id": traj_id, "score": score} for traj_id, score in topk]
    topk_path = output_dir / f"topk_{dataset}_{profile}.json"
    topk_path.write_text(json.dumps(topk_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ecpo_path = output_dir / f"ecpo_output_{dataset}_{profile}.json"
    ecpo_path.write_text(json.dumps(ecpo_output, ensure_ascii=False, indent=2), encoding="utf-8")

    reasons_path = output_dir / f"reasons_{dataset}_{profile}.jsonl"
    with reasons_path.open("w", encoding="utf-8") as f:
        for reason in reasons:
            f.write(json.dumps(reason, ensure_ascii=False) + "\n")
    LOGGER.debug("Outputs written: %s, %s, %s", topk_path, ecpo_path, reasons_path)


def _resolve_profiles(config: Dict, names: Iterable[str]) -> Dict[str, Dict]:
    profiles = config.get("profiles", {})
    resolved: Dict[str, Dict] = {}
    for name in names:
        profile_cfg = profiles.get(name, {})
        if not isinstance(profile_cfg, dict):
            raise ValueError(f"profile '{name}' must be a mapping")
        resolved[name] = profile_cfg
    return resolved


def _resolve_datasets(config: Dict, names: Iterable[str]) -> Dict[str, Dict]:
    datasets = config.get("datasets", {})
    resolved: Dict[str, Dict] = {}
    for name in names:
        dataset_cfg = datasets.get(name, {})
        if not isinstance(dataset_cfg, dict):
            raise ValueError(f"dataset '{name}' must be a mapping")
        resolved[name] = dataset_cfg
    return resolved


def _evaluate_once(root: Path, cfg: Dict, dataset: str, profile: str, force_sample: bool) -> Dict[str, float]:
    traj_path = _resolve_path(root, cfg["traj_path"])
    reward_ckpt = _resolve_path(root, cfg["reward_ckpt"])
    policy_logprobs = _resolve_path(root, cfg["policy_logprobs"])
    output_dir = _resolve_path(root, cfg["output_dir"])
    alpha = float(cfg["alpha"])
    k = int(cfg["k"])

    auto_sample = bool(cfg.get("auto_sample", False))
    missing_assets = not traj_path.exists() or not reward_ckpt.exists()
    if force_sample or (auto_sample and missing_assets):
        LOGGER.debug("Missing assets detected (traj=%s reward=%s). Generating sample data.", traj_path.exists(), reward_ckpt.exists())
        sample_paths = _ensure_sample_assets(root, cfg)
        traj_path = sample_paths["traj_path"]
        reward_ckpt = sample_paths["reward_ckpt"]
        policy_logprobs = sample_paths["policy_logprobs"]

    LOGGER.debug(
        "Evaluating dataset=%s profile=%s\n traj_path=%s\n reward_ckpt=%s\n policy_logprobs=%s\n output_dir=%s\n alpha=%.3f k=%d",
        dataset,
        profile,
        traj_path,
        reward_ckpt,
        policy_logprobs,
        output_dir,
        alpha,
        k,
    )

    trajectories = load_trajectories(traj_path)
    LOGGER.debug("Loaded %d trajectories", len(trajectories))

    combined_scores, reasons = _score_trajectories(trajectories, reward_ckpt, policy_logprobs, alpha)
    window = build_window_from_trajectories(trajectories, window_id=f"{dataset}_{profile}", intent_id="ECPO")
    ecpo_output = build_policy_output(window, [traj_id for traj_id, _score in combined_scores], k)
    metrics = evaluate_ranking(combined_scores, trajectories, k)
    certified = evaluate_certified_output(
        ecpo_output,
        window,
        ECPOValidator(k=k),
        DeterministicEvidenceVerifier(),
        k,
    )
    metrics_fmt = {
        "NDCG@10": metrics["NDCG@K"],
        "MAP": metrics["MAP@K"],
        "Hit@10": metrics["Hit@K"],
        "CertNDCG@10": certified["CertNDCG@K"],
        "EvidCons@10": certified["EvidCons@K"],
        "Feasible": certified["Feasible"],
    }
    _write_outputs(output_dir, dataset, profile, combined_scores[:k], reasons, ecpo_output)
    eval_path = output_dir / f"evaluation_{dataset}_{profile}.json"
    eval_path.write_text(json.dumps(metrics_fmt, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.debug("Metrics written: %s", eval_path)
    return metrics_fmt


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ECPO ranking and certificate metrics")
    parser.add_argument("--config", type=Path, default=Path("ecpo_rl/config.yaml"))
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--profiles", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--auto-sample", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config = _load_config(_resolve_path(root, args.config))

    base_cfg = config.get("defaults", {})
    if not isinstance(base_cfg, dict):
        raise ValueError("defaults must be a mapping")

    defaults_by_mode = config.get("defaults_by_mode", {}) if isinstance(config.get("defaults_by_mode", {}), dict) else {}
    if defaults_by_mode:
        try:
            mode, _ = resolve_mode(config)
            mode_defaults = defaults_by_mode.get(mode, {})
            if isinstance(mode_defaults, dict):
                base_cfg = _deep_merge(base_cfg, mode_defaults)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to resolve defaults_by_mode: %s", exc)

    run_cfg = config.get("run", {}) if isinstance(config.get("run", {}), dict) else {}
    dataset_spec = args.datasets or str(run_cfg.get("datasets", "MAVEN-ERE"))
    profile_spec = args.profiles or str(run_cfg.get("profiles", "base"))
    dataset_names = [name.strip() for name in dataset_spec.split(",") if name.strip()]
    profile_names = [name.strip() for name in profile_spec.split(",") if name.strip()]
    run_k = run_cfg.get("k")
    run_alpha = run_cfg.get("alpha")
    run_auto_sample = bool(run_cfg.get("auto_sample", False))

    datasets = _resolve_datasets(config, dataset_names)
    profiles = _resolve_profiles(config, profile_names)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, dataset_cfg in datasets.items():
        results.setdefault(dataset, {})
        for profile, profile_cfg in profiles.items():
            cfg = _deep_merge(base_cfg, dataset_cfg)
            cfg = _deep_merge(cfg, profile_cfg)
            if run_k is not None:
                cfg["k"] = run_k
            if run_alpha is not None:
                cfg["alpha"] = run_alpha
            if args.k is not None:
                cfg["k"] = args.k
            if args.alpha is not None:
                cfg["alpha"] = args.alpha
            force_sample = args.auto_sample or run_auto_sample
            LOGGER.debug(
                "Run config resolved dataset=%s profile=%s k=%s alpha=%s auto_sample=%s",
                dataset,
                profile,
                cfg.get("k"),
                cfg.get("alpha"),
                force_sample,
            )
            metrics = _evaluate_once(root, cfg, dataset, profile, force_sample)
            results[dataset][profile] = metrics

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
