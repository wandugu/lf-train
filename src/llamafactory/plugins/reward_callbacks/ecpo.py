# -*- coding: utf-8 -*-
"""ECPO reward callback for PPO/GRPO rollouts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ecpo_rl.ecpo import ECPORewardCalculator


LOGGER = logging.getLogger("ecpo_reward")


class ECPORankRewardCallback:
    """Score strict JSON ranking/certificate outputs with the ECPO reward."""

    def __init__(
        self,
        reward_ckpt: Path,
        trajectory_path: Optional[Path] = None,
        k: int = 10,
        gamma: float = 0.95,
        lambda_cert: float = 1.0,
        lambda_cycle: float = 1.0,
        invalid_penalty: float = 1.0,
        missing_penalty: float = 0.25,
        reward_clip: Optional[Tuple[float, float]] = None,
        support_threshold: float = 0.35,
        support_margin: float = 0.05,
        **_ignored: object,
    ) -> None:
        traj_path = Path(trajectory_path) if trajectory_path is not None else Path("data/processed/traj.jsonl")
        self.calculator = ECPORewardCalculator(
            reward_ckpt=Path(reward_ckpt),
            trajectory_path=traj_path,
            k=k,
            gamma=gamma,
            lambda_cert=lambda_cert,
            lambda_cycle=lambda_cycle,
            invalid_penalty=invalid_penalty,
            missing_penalty=missing_penalty,
            reward_clip=reward_clip,
            support_threshold=support_threshold,
            support_margin=support_margin,
        )

    def __call__(
        self,
        sequences: Sequence[Sequence[int]],
        response_scores: Optional[Sequence[float]] = None,
        metas: Optional[Sequence[Dict]] = None,
        logprobs: Optional[Sequence[float]] = None,
    ) -> List[float]:
        del response_scores, logprobs
        metas = list(metas) if metas is not None else [{} for _ in sequences]
        response_texts = [str(meta.get("response") or meta.get("raw_response") or "") for meta in metas]
        rewards = self.calculator.score_batch(metas, response_texts)
        LOGGER.debug("ECPO batch rewards: %s", rewards)
        return rewards


def build_reward_callback(**kwargs: object) -> ECPORankRewardCallback:
    return ECPORankRewardCallback(**kwargs)


__all__ = ["ECPORankRewardCallback", "build_reward_callback"]
