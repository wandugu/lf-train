from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import yaml


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config yaml must be a mapping")
    return payload


def resolve_mode(config: Dict) -> Tuple[str, Dict]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    mode = str(data_cfg.get("mode", "maven"))
    modes = data_cfg.get("modes", {}) if isinstance(data_cfg.get("modes", {}), dict) else {}
    mode_cfg = modes.get(mode)
    if not isinstance(mode_cfg, dict):
        raise ValueError(f"mode '{mode}' is not configured in data.modes")
    return mode, mode_cfg


def resolve_processed_files(config: Dict) -> Dict[str, str]:
    _, mode_cfg = resolve_mode(config)
    processed = mode_cfg.get("processed_files", {})
    if not isinstance(processed, dict):
        raise ValueError("processed_files must be a mapping")
    return {key: str(value) for key, value in processed.items()}


def resolve_mode_json(path: Path) -> str:
    config = load_config(path)
    mode, mode_cfg = resolve_mode(config)
    payload = {"mode": mode, "config": mode_cfg}
    return json.dumps(payload, ensure_ascii=False, indent=2)
