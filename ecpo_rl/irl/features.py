# -*- coding: utf-8 -*-
"""Feature engineering utilities for ECPO trajectory reward learning."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from tgn_encoder import SimpleTemporalGraphEncoder  # type: ignore
else:
    from .tgn_encoder import SimpleTemporalGraphEncoder


def build_skeleton_features(trajectory: Dict) -> np.ndarray:
    """Compute skeleton-level features.

    The feature vector encodes:

    1. mean skeleton hits per step
    2. coverage of canonical phases (PREP, PROBE, EXECUTE, OUTCOME)
    3. penalty for missing EXECUTE phase
    4. temporal coherence penalty (based on delta days)
    """

    steps: List[Dict] = trajectory.get("steps", [])
    assert steps, "trajectory must contain steps"

    skeleton_counts = [len(step.get("skeleton_hits", [])) for step in steps]
    mean_hits = float(np.mean(skeleton_counts))

    canonical = ["PREP", "PROBE", "EXECUTE", "OUTCOME"]
    hits = {phase: 0 for phase in canonical}
    for step in steps:
        for token in step.get("skeleton_hits", []):
            stage = "OUTCOME" if token == "CASHOUT" else token
            if stage in hits:
                hits[stage] += 1
    coverage = sum(1 for v in hits.values() if v > 0) / len(canonical)
    missing_execute = 1.0 if hits["EXECUTE"] == 0 else 0.0

    deltas = [abs(step.get("delta_days_from_prev", 0)) for step in steps[1:]]
    temporal_penalty = float(np.mean(deltas)) if deltas else 0.0

    return np.array([mean_hits, coverage, missing_execute, temporal_penalty], dtype=np.float32)


@lru_cache(maxsize=1)
def _get_tgn_encoder() -> SimpleTemporalGraphEncoder:
    return SimpleTemporalGraphEncoder(node_embedding_dim=16, time_decay=0.1)


def build_tgn_features(trajectory: Dict, graph: Dict) -> np.ndarray:
    """Encode graph-temporal statistics via a tiny TGN encoder."""

    encoder = _get_tgn_encoder()
    nodes = graph.get("graph_nodes", [])
    edges = graph.get("graph_edges", [])
    if not nodes:
        raise AssertionError("graph must contain nodes")

    node_embeddings = encoder(nodes, edges)
    agg = node_embeddings.mean(axis=0)
    stats = np.array([
        float(np.max(node_embeddings)),
        float(np.min(node_embeddings)),
        float(np.linalg.norm(agg)),
    ], dtype=np.float32)
    return stats


def build_consistency_features(trajectory: Dict) -> np.ndarray:
    steps: List[Dict] = trajectory.get("steps", [])
    assert steps, "trajectory must contain steps"

    # Role consistency – same agent across steps
    agent_ids = [step.get("roles", {}).get("Agent") for step in steps if step.get("roles")]
    unique_agents = len(set(agent_ids)) if agent_ids else 0
    agent_consistency = 1.0 / unique_agents if unique_agents else 0.0

    # Target persistence ratio
    target_ids = [step.get("roles", {}).get("Target") for step in steps if step.get("roles")]
    target_consistency = 1.0
    if target_ids:
        target_consistency = sum(t == target_ids[0] for t in target_ids) / len(target_ids)

    # Chronological order penalty
    times = [step.get("time") for step in steps]
    gaps = []
    for prev, cur in zip(times, times[1:]):
        if prev and cur and prev <= cur:
            gaps.append(0)
        else:
            gaps.append(1)
    chronology_penalty = float(np.mean(gaps)) if gaps else 0.0

    return np.array([agent_consistency, target_consistency, chronology_penalty], dtype=np.float32)


def build_feature_vector(trajectory: Dict) -> np.ndarray:
    skeleton = build_skeleton_features(trajectory)
    tgn = build_tgn_features(trajectory, trajectory.get("meta", {}))
    consistency = build_consistency_features(trajectory)
    return np.concatenate([skeleton, tgn, consistency], axis=0)


def _test_features() -> None:
    sample = {
        "steps": [
            {
                "skeleton_hits": ["PREP", "PROBE"],
                "delta_days_from_prev": 0,
                "roles": {"Agent": "A", "Target": "T"},
                "time": "2014-03-20",
            },
            {
                "skeleton_hits": ["EXECUTE"],
                "delta_days_from_prev": 1,
                "roles": {"Agent": "A", "Target": "T"},
                "time": "2014-03-21",
            },
        ],
        "meta": {"graph_nodes": ["A", "T"], "graph_edges": [["A", "EXECUTE", "T", "2014-03-21"]]},
    }
    vec = build_feature_vector(sample)
    assert vec.shape[0] == 10, f"unexpected feature size {vec.shape}"
    assert vec[0] > 0 and vec[1] > 0, "skeleton features should be positive"


if __name__ == "__main__":
    _test_features()
    print("Feature tests passed.")
