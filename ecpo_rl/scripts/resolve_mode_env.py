#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""输出当前 mode 的环境变量，用于 shell 脚本读取配置。"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Dict

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ecpo_rl.config_utils import load_config, resolve_mode, resolve_processed_files


def _quote(value: str) -> str:
    return shlex.quote(value)


def _emit(env: Dict[str, str]) -> None:
    for key, value in env.items():
        print(f"{key}={_quote(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve ECPO mode configs for shell scripts")
    parser.add_argument("--config", type=Path, default=Path("ecpo_rl/config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    mode, mode_cfg = resolve_mode(config)
    processed_files = resolve_processed_files(config)

    train_cfg = mode_cfg.get("train", {}) if isinstance(mode_cfg.get("train", {}), dict) else {}
    wandb_cfg = mode_cfg.get("wandb", {}) if isinstance(mode_cfg.get("wandb", {}), dict) else {}
    dataset_info = mode_cfg.get("dataset_info", {}) if isinstance(mode_cfg.get("dataset_info", {}), dict) else {}

    env = {
        "ECPO_MODE": mode,
        "ECPO_PROCESSED_DIR": str(mode_cfg.get("processed_dir", "data/processed")),
        "ECPO_EVENT_FILE": processed_files.get("event", "event.jsonl"),
        "ECPO_TRAJ_FILE": processed_files.get("traj", "traj.jsonl"),
        "ECPO_PAIRS_FILE": processed_files.get("pairs", "pairs.jsonl"),
        "ECPO_SFT_FILE": processed_files.get("sft", "maven_sft.jsonl"),
        "ECPO_RL_PROMPTS_FILE": processed_files.get("rl_prompts", "rl_prompts.jsonl"),
        "ECPO_SFT_DATASET": str(dataset_info.get("sft", "maven_sft")),
        "ECPO_RL_DATASET": str(dataset_info.get("rl", "maven_rl")),
        "ECPO_PRETRAIN_CONFIG": str(train_cfg.get("pretrain_config", "configs/pretrain_maven.yaml")),
        "ECPO_EXPORT_CONFIG": str(train_cfg.get("export_config", "configs/export_qwen_maven_sft.yaml")),
        "ECPO_PPO_CONFIG": str(train_cfg.get("ppo_config", "configs/ppo_rl.yaml")),
        "ECPO_REWARD_OUTPUT_DIR": str(train_cfg.get("reward_output_dir", "outputs/qwen-4b-rm")),
        "ECPO_REWARD_CKPT": str(train_cfg.get("reward_ckpt", "outputs/qwen-4b-rm/reward.ckpt")),
        "ECPO_POLICY_OUTPUT_DIR": str(train_cfg.get("policy_output_dir", "outputs/qwen-4b-rl")),
        "ECPO_WANDB_PROJECT": str(wandb_cfg.get("project", "maven-ecpo")),
        "ECPO_WANDB_SFT_TAGS": str(wandb_cfg.get("sft_tags", "sft")),
        "ECPO_WANDB_PPO_TAGS": str(wandb_cfg.get("ppo_tags", "ecpo")),
    }

    _emit(env)


if __name__ == "__main__":
    main()
