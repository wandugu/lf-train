# -*- coding: utf-8 -*-
"""Fallback ECPO trainer for environments without PPO/GRPO support.

This module simulates the ECPO reward loop so that the demo can run fully
offline.  When LlamaFactory provides PPO/GRPO, prefer invoking the official CLI
with ``configs/ppo_rl.yaml`` instead of this script.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ecpo_rl.ecpo import build_policy_output  # type: ignore
    from ecpo_rl.ecpo.reward import ECPORewardCalculator  # type: ignore
else:
    from ..ecpo import build_policy_output
    from ..ecpo.reward import ECPORewardCalculator


LOGGER = logging.getLogger(__name__)


class HeuristicRLTrainer:
    def __init__(
        self,
        reward_ckpt: Path,
        trajectory_path: Path,
        prompts_path: Path,
        output_dir: Path,
        alpha: float = 0.6,
        k: int = 10,
    ) -> None:
        del alpha
        self.reward_model = ECPORewardCalculator(
            reward_ckpt=reward_ckpt,
            trajectory_path=trajectory_path,
            k=k,
        )
        self.prompts_path = prompts_path
        self.output_dir = output_dir
        self.k = k

    def load_prompts(self) -> List[Dict]:
        prompts: List[Dict] = []
        with self.prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line))
        if not prompts:
            raise ValueError("no RL prompts found")
        return prompts

    def simulate_policy(self, prompt: Dict) -> Dict[str, float]:
        meta = dict(prompt.get("_meta") or {})
        for key in ("trajectory_id", "person_id", "window_id", "intent_id", "candidate_ids", "candidate_map", "skeleton_steps"):
            if key in prompt and key not in meta:
                meta[key] = prompt[key]

        try:
            window = self.reward_model.build_window_for_meta(meta)
            ranked_ids = list(window.candidate_ids)
            response = json.dumps(build_policy_output(window, ranked_ids, self.k), ensure_ascii=False)
            reward, details = self.reward_model.score_response(response, meta)
            LOGGER.debug("offline ECPO details: %s", details)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("offline ECPO simulation failed for %s: %s", prompt.get("trajectory_id"), exc)
            reward = 0.0
        return {"reward": reward, "logprob": float(np.tanh(reward))}

    def train(self) -> None:
        prompts = self.load_prompts()
        stats = []
        policy_logprobs = {}
        for prompt in prompts:
            prompt_text = str(prompt.get("prompt", "")).strip()
            response_text = str(prompt.get("response", "")).strip()
            trajectory_id = prompt.get("trajectory_id") or "<unknown>"
            LOGGER.info(
                "RL 样本 %s\n[Prompt]\n%s\n[Response]\n%s",
                trajectory_id,
                prompt_text or "<empty>",
                response_text or "<empty>",
            )
            simulation = self.simulate_policy(prompt)
            combined = simulation["reward"]
            policy_logprobs[prompt["trajectory_id"]] = combined
            stats.append(combined)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "policy_logprobs.json").write_text(json.dumps(policy_logprobs, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {"mean_logprob": float(np.mean(stats)), "num_prompts": len(stats)}
        (self.output_dir / "heuristic_rl.stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heuristic ECPO trainer (offline demo)")
    parser.add_argument("--reward-ckpt", type=Path, required=True)
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--k", type=int, default=10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    trainer = HeuristicRLTrainer(
        args.reward_ckpt,
        args.trajectory_path,
        args.prompts,
        args.output_dir,
        alpha=args.alpha,
        k=args.k,
    )
    trainer.train()


if __name__ == "__main__":
    main()
