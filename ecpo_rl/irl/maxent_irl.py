# -*- coding: utf-8 -*-
"""Skeleton-conditioned trajectory reward trainer for ECPO."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np

try:  # pragma: no cover - 可选依赖
    import wandb
except Exception:  # noqa: BLE001 - 尽量保证离线环境也能运行
    wandb = None  # type: ignore[assignment]


def _wandb_run() -> "wandb.sdk.wandb_run.Run | None":  # type: ignore[name-defined]
    """延迟初始化 wandb，确保在无依赖环境下也能运行。"""

    if wandb is None:  # type: ignore[truthy-function]
        return None

    run = getattr(_wandb_run, "_cached_run", None)
    if run is not None:
        return run

    try:
        run = wandb.init(  # type: ignore[attr-defined]
            project="maven-ecpo",
            config={
                "learning_rate": 0.05,
                "epochs": 150,
                "batch_size": 32,
            },
            name="maven-rm",
            tags=["reward", "ecpo", "qwen3-4b", "maven"],
        )
    except Exception:  # noqa: BLE001 - wandb 初始化失败时静默降级
        run = None

    setattr(_wandb_run, "_cached_run", run)
    return run


def _wandb_log(payload: dict[str, float]) -> None:
    run = _wandb_run()
    if run is None:
        return

    try:
        wandb.log(payload)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _wandb_save(path: str) -> None:
    run = _wandb_run()
    if run is None:
        return

    try:
        wandb.save(path)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _wandb_finish() -> None:
    run = getattr(_wandb_run, "_cached_run", None)
    if run is None:
        return

    try:
        wandb.finish()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from features import build_feature_vector  # type: ignore
else:
    from .features import build_feature_vector


@dataclass
class Trajectory:
    raw: Dict
    features: np.ndarray
    label: str

    @property
    def trajectory_id(self) -> str:
        return self.raw.get("trajectory_id", "unknown")


class MaxEntIRL:
    def __init__(self, temperature: float = 1.0, lr: float = 0.1, epochs: int = 300) -> None:
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.theta = None

    def load_trajectories(self, path: Path) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                features = build_feature_vector(raw)
                label = raw.get("label", "candidate")
                trajectories.append(Trajectory(raw=raw, features=features, label=label))
        if not trajectories:
            raise ValueError("empty trajectory file")
        feature_dims = {traj.features.shape[0] for traj in trajectories}
        assert len(feature_dims) == 1, "all trajectories must share feature dimensionality"
        return trajectories

    def load_pairs(self, path: Path) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        if not path.exists():
            return pairs
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                pairs.append(pair)
        return pairs

    def _data_expectation(self, trajectories: Sequence[Trajectory]) -> np.ndarray:
        expert = [traj.features for traj in trajectories if traj.label == "expert"]
        if not expert:
            expert = [traj.features for traj in trajectories]
        return np.mean(expert, axis=0)

    def _model_expectation(self, weights: np.ndarray, trajectories: Sequence[Trajectory]) -> np.ndarray:
        logits = np.array([np.dot(weights, traj.features) / self.temperature for traj in trajectories])
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        expectation = sum(p * traj.features for p, traj in zip(probs, trajectories))
        return expectation

    def _pairwise_loss(self, weights: np.ndarray, trajectories: Sequence[Trajectory], pairs: Sequence[Dict[str, str]]) -> float:
        if not pairs:
            return 0.0
        traj_map = {traj.trajectory_id: traj for traj in trajectories}
        loss = 0.0
        for pair in pairs:
            better = traj_map.get(pair["better"])
            worse = traj_map.get(pair["worse"])
            if not better or not worse:
                continue
            margin = float(np.dot(weights, better.features - worse.features))
            loss += np.log(1 + np.exp(-margin))
        return loss / max(len(pairs), 1)

    def fit(self, trajectories: Sequence[Trajectory], pairs: Sequence[Dict[str, str]]) -> None:
        feature_dim = trajectories[0].features.shape[0]
        weights = np.zeros(feature_dim, dtype=np.float32)
        data_expectation = self._data_expectation(trajectories)

        for epoch in range(1, self.epochs + 1):
            model_expectation = self._model_expectation(weights, trajectories)
            grad = data_expectation - model_expectation
            weights += self.lr * grad

            # 记录训练损失到 wandb
            _wandb_log(
                {
                    "epoch": epoch,
                    "loss": float(np.linalg.norm(data_expectation - model_expectation)),
                }
            )

            # pairwise regulariser gradient (approximation)
            if pairs:
                traj_map = {traj.trajectory_id: traj for traj in trajectories}
                pair_grad = np.zeros_like(weights)
                for pair in pairs:
                    better = traj_map.get(pair["better"])
                    worse = traj_map.get(pair["worse"])
                    if not better or not worse:
                        continue
                    diff = better.features - worse.features
                    margin = float(np.dot(weights, diff))
                    prob = 1.0 / (1.0 + np.exp(margin))
                    pair_grad += prob * diff
                pair_grad /= max(len(pairs), 1)
                weights += self.lr * 0.1 * pair_grad

            if epoch % 50 == 0:
                loss = np.linalg.norm(data_expectation - model_expectation)
                print(f"[IRL] epoch={epoch} loss={loss:.4f}")

        self.theta = weights

    def save(self, path: Path) -> None:
        if self.theta is None:
            raise ValueError("model not trained")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"theta": self.theta.tolist(), "temperature": self.temperature}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # 上传模型文件到 wandb（若可用）
        _wandb_save(str(path))

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text("utf-8"))
        self.theta = np.array(payload["theta"], dtype=np.float32)
        self.temperature = payload.get("temperature", 1.0)

    def score(self, trajectory: Dict) -> float:
        if self.theta is None:
            raise ValueError("model not initialised")
        features = build_feature_vector(trajectory)
        return float(np.dot(self.theta, features))


def train(args: argparse.Namespace) -> None:
    trainer = MaxEntIRL(temperature=args.temperature, lr=args.lr, epochs=args.epochs)
    trajectories = trainer.load_trajectories(args.traj_path)
    pairs = trainer.load_pairs(args.pair_path)
    trainer.fit(trajectories, pairs)
    trainer.save(args.output)

    stats = {
        "num_traj": len(trajectories),
        "feature_dim": trajectories[0].features.shape[0],
        "pairs": len(pairs),
    }
    args.output.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ECPO trajectory reward model")
    parser.add_argument("--traj-path", type=Path, required=True)
    parser.add_argument("--pair-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    return parser


def _test_scoring(tmp_dir: Path) -> None:
    traj_path = tmp_dir / "traj.jsonl"
    pair_path = tmp_dir / "pairs.jsonl"
    traj_payload = {
        "person_id": "P",
        "trajectory_id": "traj_1",
        "label": "expert",
        "steps": [
            {
                "skeleton_hits": ["PREP", "EXECUTE"],
                "delta_days_from_prev": 0,
                "roles": {"Agent": "A", "Target": "T"},
                "time": "2014-03-21",
            }
        ],
        "meta": {"graph_nodes": ["A", "T"], "graph_edges": [["A", "EXECUTE", "T", "2014-03-21"]]},
    }
    traj_path.write_text(json.dumps(traj_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    pair_path.write_text(json.dumps({"better": "traj_1", "worse": "traj_1", "reason": ""}, ensure_ascii=False) + "\n", encoding="utf-8")

    args = build_argparser().parse_args(
        ["--traj-path", str(traj_path), "--pair-path", str(pair_path), "--output", str(tmp_dir / "reward.ckpt"), "--epochs", "5"]
    )
    train(args)
    model = MaxEntIRL()
    model.load(tmp_dir / "reward.ckpt")
    score = model.score(traj_payload)
    assert isinstance(score, float)


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
    # 训练完成后，关闭 wandb 记录
    _wandb_finish()
