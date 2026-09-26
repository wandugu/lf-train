# -*- coding: utf-8 -*-
"""ECPO window validation, evidence-only recovery, and reward shaping."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ecpo_rl.irl.maxent_irl import MaxEntIRL


LOGGER = logging.getLogger(__name__)

ECPO_STAGES: Tuple[str, ...] = ("PREP", "PROBE", "EXECUTE", "OUTCOME")
STAGE_ALIASES = {"CASHOUT": "OUTCOME"}
LABEL_RELEVANCE = {"expert": 2, "candidate": 1, "negative": 0}


def normalize_stage(value: Any) -> str:
    stage = str(value or "").strip().upper()
    return STAGE_ALIASES.get(stage, stage)


def _candidate_id(trajectory: Dict[str, Any]) -> str:
    return str(
        trajectory.get("candidate_id")
        or trajectory.get("trajectory_id")
        or trajectory.get("person_id")
        or "unknown"
    )


def _trajectory_id(trajectory: Dict[str, Any]) -> str:
    return str(trajectory.get("trajectory_id") or _candidate_id(trajectory))


def _normalise_skeleton_steps(value: Any = None) -> List[str]:
    if not value:
        return list(ECPO_STAGES)

    stages: List[str] = []
    if isinstance(value, str):
        raw_items = [item.strip() for item in re.split(r"[,\s>/-]+", value) if item.strip()]
    elif isinstance(value, Sequence):
        raw_items = list(value)
    else:
        raw_items = []

    for item in raw_items:
        if isinstance(item, dict):
            stage = normalize_stage(item.get("etype") or item.get("step_id") or item.get("stage"))
        else:
            stage = normalize_stage(item)
        if stage and stage not in stages:
            stages.append(stage)

    return stages or list(ECPO_STAGES)


def parse_span(value: Any, *, allow_point: bool = False) -> Optional[Tuple[int, int]]:
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
        if not match:
            return None
        start, end = int(match.group(1)), int(match.group(2))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)) and len(value) == 2:
        try:
            start, end = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    else:
        return None

    if allow_point and end == start:
        end += 1
    if start < 0 or end <= start:
        return None
    return start, end


def _span_overlaps(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _step_stages(step: Dict[str, Any]) -> List[str]:
    stages = [normalize_stage(item) for item in step.get("skeleton_hits", [])]
    return [stage for stage in stages if stage]


def _step_matches_stage(step: Dict[str, Any], stage: str) -> bool:
    return normalize_stage(stage) in _step_stages(step)


def _iter_step_refs(step: Dict[str, Any]) -> Iterable[Tuple[str, Tuple[int, int]]]:
    for ref in step.get("text_refs", []):
        if not isinstance(ref, dict):
            continue
        doc_id = ref.get("doc_id")
        span = parse_span(ref.get("span"), allow_point=True)
        if isinstance(doc_id, str) and span is not None:
            yield doc_id, span


def _find_step_by_event_id(trajectory: Dict[str, Any], event_id: str) -> Optional[Dict[str, Any]]:
    for step in trajectory.get("steps", []):
        if str(step.get("event_id")) == event_id:
            return step
    return None


@dataclass
class ECPOWindow:
    window_id: str
    intent_id: str
    candidate_ids: List[str]
    trajectories: Dict[str, Dict[str, Any]]
    skeleton_steps: List[str] = field(default_factory=lambda: list(ECPO_STAGES))
    doc_ids: List[str] = field(default_factory=list)
    doc_lengths: Dict[str, int] = field(default_factory=dict)

    def top_k(self, requested_k: int) -> int:
        return max(0, min(int(requested_k), len(self.candidate_ids)))


@dataclass
class BundleValidation:
    candidate_id: str
    valid: bool
    step_coverage: float
    traceability: float
    role_support: float
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    parsed: bool
    valid: bool
    topk: List[str]
    errors: List[str]
    bundle_results: List[BundleValidation]
    expected_k: int

    @property
    def missing(self) -> int:
        return max(0, self.expected_k - len(self.topk))

    @property
    def cert_reward(self) -> float:
        if self.expected_k <= 0:
            return 0.0
        covered = 0.0
        for result in self.bundle_results[: self.expected_k]:
            covered += result.step_coverage if result.valid else 0.0
        return covered / self.expected_k


def build_window_from_trajectories(
    trajectories: Sequence[Dict[str, Any]],
    *,
    window_id: str = "window_default",
    intent_id: str = "default",
    skeleton_steps: Any = None,
) -> ECPOWindow:
    trajectory_map: Dict[str, Dict[str, Any]] = {}
    doc_lengths: Dict[str, int] = {}
    doc_ids: List[str] = []

    for raw in trajectories:
        if not isinstance(raw, dict):
            continue
        traj = dict(raw)
        candidate_id = _candidate_id(traj)
        traj["candidate_id"] = candidate_id
        trajectory_map[candidate_id] = traj
        for step in traj.get("steps", []):
            for doc_id, span in _iter_step_refs(step):
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                doc_lengths[doc_id] = max(doc_lengths.get(doc_id, 0), span[1])

    candidate_ids = list(trajectory_map.keys())
    return ECPOWindow(
        window_id=str(window_id),
        intent_id=str(intent_id),
        candidate_ids=candidate_ids,
        trajectories=trajectory_map,
        skeleton_steps=_normalise_skeleton_steps(skeleton_steps),
        doc_ids=doc_ids,
        doc_lengths=doc_lengths,
    )


def load_trajectories(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_certificate_for_trajectory(
    trajectory: Dict[str, Any],
    skeleton_steps: Sequence[str],
) -> Dict[str, Any]:
    used_events: set[str] = set()
    cert_steps: List[Dict[str, Any]] = []

    for stage in skeleton_steps:
        stage = normalize_stage(stage)
        matched_step: Optional[Dict[str, Any]] = None
        for step in trajectory.get("steps", []):
            event_id = str(step.get("event_id"))
            if event_id in used_events:
                continue
            if _step_matches_stage(step, stage):
                matched_step = step
                used_events.add(event_id)
                break

        if matched_step is None:
            cert_steps.append(
                {"step_id": stage, "etype": stage, "matched": False, "event_id": None, "evidence": []}
            )
            continue

        evidence = []
        for doc_id, span in _iter_step_refs(matched_step):
            evidence.append({"doc_id": doc_id, "span": [span[0], span[1]], "kind": "trigger"})

        if evidence:
            cert_steps.append(
                {
                    "step_id": stage,
                    "etype": stage,
                    "matched": True,
                    "event_id": matched_step.get("event_id"),
                    "evidence": evidence,
                }
            )
        else:
            cert_steps.append(
                {"step_id": stage, "etype": stage, "matched": False, "event_id": None, "evidence": []}
            )

    return {"steps": cert_steps}


def build_policy_output(window: ECPOWindow, ranked_candidate_ids: Sequence[str], k: int) -> Dict[str, Any]:
    kw = window.top_k(k)
    topk = [str(candidate_id) for candidate_id in ranked_candidate_ids if str(candidate_id) in window.trajectories]
    topk = topk[:kw]
    certificates = [
        build_certificate_for_trajectory(window.trajectories[candidate_id], window.skeleton_steps)
        for candidate_id in topk
    ]
    return {"window_id": window.window_id, "topk": topk, "certificates": certificates}


def parse_policy_output(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    raw = (text or "").strip()
    if not raw:
        return None, False
    try:
        payload = json.loads(raw)
        return (payload if isinstance(payload, dict) else None), False
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            payload = json.loads(fence.group(1))
            return (payload if isinstance(payload, dict) else None), True
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        try:
            payload = json.loads(raw[start : end + 1])
            return (payload if isinstance(payload, dict) else None), True
        except json.JSONDecodeError:
            return None, False
    return None, False


class ECPOValidator:
    def __init__(self, k: int = 10) -> None:
        self.k = int(k)

    def validate(self, payload: Optional[Dict[str, Any]], window: ECPOWindow) -> ValidationResult:
        expected_k = window.top_k(self.k)
        if payload is None:
            return ValidationResult(
                parsed=False,
                valid=False,
                topk=[],
                errors=["output is not parseable JSON object"],
                bundle_results=[],
                expected_k=expected_k,
            )

        errors: List[str] = []
        if payload.get("window_id") != window.window_id:
            errors.append("window_id mismatch")

        raw_topk = payload.get("topk")
        if not isinstance(raw_topk, list):
            raw_topk = []
            errors.append("topk must be a list")
        topk = [str(item) for item in raw_topk if isinstance(item, str)]
        if len(topk) != len(raw_topk):
            errors.append("topk must contain only string candidate ids")
        if len(topk) != expected_k:
            errors.append(f"topk length must be {expected_k}, got {len(topk)}")
        if len(set(topk)) != len(topk):
            errors.append("topk contains duplicate candidate ids")
        for candidate_id in topk:
            if candidate_id not in window.candidate_ids:
                errors.append(f"candidate id not in roster: {candidate_id}")

        certificates = payload.get("certificates")
        if not isinstance(certificates, list):
            certificates = []
            errors.append("certificates must be a list")
        if len(certificates) != len(topk):
            errors.append("certificates length must match topk length")

        bundle_results: List[BundleValidation] = []
        for index, candidate_id in enumerate(topk[: min(len(topk), len(certificates), expected_k)]):
            bundle = certificates[index]
            result = self._validate_bundle(bundle, candidate_id, window, index)
            bundle_results.append(result)
            errors.extend(f"cert[{index}]: {error}" for error in result.errors)

        valid = not errors and len(bundle_results) == expected_k and all(result.valid for result in bundle_results)
        return ValidationResult(
            parsed=True,
            valid=valid,
            topk=topk,
            errors=errors,
            bundle_results=bundle_results,
            expected_k=expected_k,
        )

    def _validate_bundle(
        self,
        bundle: Any,
        candidate_id: str,
        window: ECPOWindow,
        index: int,
    ) -> BundleValidation:
        del index
        errors: List[str] = []
        trajectory = window.trajectories.get(candidate_id)
        if trajectory is None:
            return BundleValidation(candidate_id, False, 0.0, 0.0, 0.0, ["missing candidate trajectory"])

        if not isinstance(bundle, dict):
            return BundleValidation(candidate_id, False, 0.0, 0.0, 0.0, ["certificate must be an object"])

        steps = bundle.get("steps")
        if not isinstance(steps, list):
            steps = []
            errors.append("steps must be a list")
        if len(steps) != len(window.skeleton_steps):
            errors.append("certificate must contain one step per skeleton step")

        covered = 0
        total_evidence = 0
        valid_evidence = 0
        role_checks = 0
        role_hits = 0

        for pos, expected_stage in enumerate(window.skeleton_steps):
            if pos >= len(steps) or not isinstance(steps[pos], dict):
                errors.append(f"missing step for {expected_stage}")
                continue

            step_obj = steps[pos]
            step_id = normalize_stage(step_obj.get("step_id"))
            etype = normalize_stage(step_obj.get("etype"))
            if step_id != expected_stage:
                errors.append(f"step_id must be {expected_stage}, got {step_obj.get('step_id')}")
            if etype != expected_stage:
                errors.append(f"etype must be {expected_stage}, got {step_obj.get('etype')}")

            matched = step_obj.get("matched")
            if not isinstance(matched, bool):
                errors.append(f"{expected_stage} matched must be boolean")
                matched = False

            evidence = step_obj.get("evidence")
            if not isinstance(evidence, list):
                errors.append(f"{expected_stage} evidence must be a list")
                evidence = []

            event_id = step_obj.get("event_id")
            if not matched:
                if event_id is not None:
                    errors.append(f"{expected_stage} unmatched step must have null event_id")
                if evidence:
                    errors.append(f"{expected_stage} unmatched step must have empty evidence")
                continue

            if not isinstance(event_id, str) or not event_id:
                errors.append(f"{expected_stage} matched step must have event_id")
                continue
            event_step = _find_step_by_event_id(trajectory, event_id)
            if event_step is None:
                errors.append(f"{expected_stage} event_id is not in the rank-aligned trajectory")
                continue
            if not _step_matches_stage(event_step, expected_stage):
                errors.append(f"{expected_stage} event_id is not skeleton-compatible")
            if not evidence:
                errors.append(f"{expected_stage} matched step must cite evidence")
                continue

            step_has_valid_evidence = False
            for item in evidence:
                total_evidence += 1
                if self._validate_evidence_item(item, event_step, expected_stage, window, errors):
                    valid_evidence += 1
                    step_has_valid_evidence = True
                    if isinstance(item, dict) and item.get("kind") == "arg":
                        role_checks += 1
                        role_hits += 1
                elif isinstance(item, dict) and item.get("kind") == "arg":
                    role_checks += 1

            if step_has_valid_evidence:
                covered += 1

        denom = max(len(window.skeleton_steps), 1)
        traceability = valid_evidence / max(total_evidence, 1)
        role_support = role_hits / role_checks if role_checks else 1.0
        return BundleValidation(
            candidate_id=candidate_id,
            valid=not errors,
            step_coverage=covered / denom,
            traceability=traceability,
            role_support=role_support,
            errors=errors,
        )

    def _validate_evidence_item(
        self,
        item: Any,
        event_step: Dict[str, Any],
        stage: str,
        window: ECPOWindow,
        errors: List[str],
    ) -> bool:
        if not isinstance(item, dict):
            errors.append(f"{stage} evidence item must be an object")
            return False

        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in window.doc_ids:
            errors.append(f"{stage} evidence doc_id is outside the window")
            return False

        span = parse_span(item.get("span"), allow_point=False)
        if span is None:
            errors.append(f"{stage} evidence span must be a non-empty half-open span")
            return False
        if span[1] > window.doc_lengths.get(doc_id, span[1]):
            errors.append(f"{stage} evidence span exceeds document bounds")
            return False

        kind = item.get("kind")
        if kind not in {"trigger", "arg"}:
            errors.append(f"{stage} evidence kind must be trigger or arg")
            return False

        overlaps_trace = any(doc_id == ref_doc and _span_overlaps(span, ref_span) for ref_doc, ref_span in _iter_step_refs(event_step))
        if not overlaps_trace:
            errors.append(f"{stage} evidence span is not traceable to the event")
            return False

        if kind == "arg":
            role = item.get("role")
            roles = event_step.get("roles", {})
            if not isinstance(role, str) or role not in roles:
                errors.append(f"{stage} argument evidence role is not supported")
                return False

        return True


class DeterministicEvidenceVerifier:
    def __init__(
        self,
        support_threshold: float = 0.35,
        support_margin: float = 0.05,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.support_threshold = float(support_threshold)
        self.support_margin = float(support_margin)
        self.weights = {
            "coverage": 0.35,
            "role": 0.15,
            "trace": 0.35,
            "precedence": 0.10,
            "bad": 0.20,
        }
        if weights:
            self.weights.update({str(key): float(value) for key, value in weights.items()})

    def score_bundle_candidate(self, bundle: Any, candidate_id: str, window: ECPOWindow) -> float:
        trajectory = window.trajectories.get(candidate_id)
        if trajectory is None or not isinstance(bundle, dict):
            return 0.0

        steps = bundle.get("steps")
        if not isinstance(steps, list):
            return 0.0

        total_evidence = 0
        trace_hits = 0
        bad = 0
        role_total = 0
        role_hits = 0
        supported_stages: Dict[str, int] = {}
        seen_refs: set[Tuple[str, Tuple[int, int], str]] = set()

        for cert_step in steps:
            if not isinstance(cert_step, dict):
                bad += 1
                continue
            stage = normalize_stage(cert_step.get("etype") or cert_step.get("step_id"))
            evidence = cert_step.get("evidence")
            if not isinstance(evidence, list):
                bad += 1
                continue

            for item in evidence:
                total_evidence += 1
                if not isinstance(item, dict):
                    bad += 1
                    continue
                doc_id = item.get("doc_id")
                span = parse_span(item.get("span"), allow_point=False)
                kind = item.get("kind")
                if not isinstance(doc_id, str) or span is None or kind not in {"trigger", "arg"}:
                    bad += 1
                    continue
                ref_key = (doc_id, span, str(kind))
                if ref_key in seen_refs:
                    bad += 1
                    continue
                seen_refs.add(ref_key)

                matched_pos = self._best_matching_step_pos(trajectory, stage, doc_id, span)
                if matched_pos is None:
                    bad += 1
                    continue

                trace_hits += 1
                supported_stages.setdefault(stage, matched_pos)
                if kind == "arg":
                    role_total += 1
                    role = item.get("role")
                    event_step = trajectory.get("steps", [])[matched_pos]
                    if isinstance(role, str) and role in event_step.get("roles", {}):
                        role_hits += 1

        coverage = len(supported_stages) / max(len(window.skeleton_steps), 1)
        trace = trace_hits / max(total_evidence, 1)
        role = role_hits / role_total if role_total else 0.0
        precedence = self._precedence_score(supported_stages, window.skeleton_steps)
        bad_ratio = bad / max(total_evidence + bad, 1)
        score = (
            self.weights["coverage"] * coverage
            + self.weights["role"] * role
            + self.weights["trace"] * trace
            + self.weights["precedence"] * precedence
            - self.weights["bad"] * bad_ratio
        )
        return float(score)

    def _best_matching_step_pos(
        self,
        trajectory: Dict[str, Any],
        stage: str,
        doc_id: str,
        span: Tuple[int, int],
    ) -> Optional[int]:
        for pos, step in enumerate(trajectory.get("steps", [])):
            if stage and not _step_matches_stage(step, stage):
                continue
            for ref_doc, ref_span in _iter_step_refs(step):
                if doc_id == ref_doc and _span_overlaps(span, ref_span):
                    return pos
        return None

    @staticmethod
    def _precedence_score(supported_stages: Dict[str, int], skeleton_steps: Sequence[str]) -> float:
        ordered = [(stage, supported_stages[stage]) for stage in skeleton_steps if stage in supported_stages]
        if len(ordered) <= 1:
            return 1.0 if ordered else 0.0
        comparisons = 0
        hits = 0
        for left, right in zip(ordered, ordered[1:]):
            comparisons += 1
            if left[1] <= right[1]:
                hits += 1
        return hits / max(comparisons, 1)

    def recover_candidates(self, certificates: Sequence[Any], window: ECPOWindow) -> List[Optional[str]]:
        if not certificates or not window.candidate_ids:
            return [None for _ in certificates]

        score_matrix = np.array(
            [
                [self.score_bundle_candidate(bundle, candidate_id, window) for candidate_id in window.candidate_ids]
                for bundle in certificates
            ],
            dtype=np.float32,
        )
        if score_matrix.size == 0:
            return [None for _ in certificates]

        assignments = self._assign(score_matrix)
        recovered: List[Optional[str]] = [None for _ in certificates]
        for row, col in assignments:
            if row >= len(certificates) or col >= len(window.candidate_ids):
                continue
            row_scores = score_matrix[row]
            best_score = float(row_scores[col])
            second_best = float(np.max(np.delete(row_scores, col))) if row_scores.size > 1 else 0.0
            if best_score >= self.support_threshold and best_score - second_best >= self.support_margin:
                recovered[row] = window.candidate_ids[col]
        return recovered

    @staticmethod
    def _assign(score_matrix: np.ndarray) -> List[Tuple[int, int]]:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rows, cols = linear_sum_assignment(-score_matrix)
            return list(zip(rows.tolist(), cols.tolist()))
        except Exception:  # noqa: BLE001
            pairs: List[Tuple[float, int, int]] = []
            for row in range(score_matrix.shape[0]):
                for col in range(score_matrix.shape[1]):
                    pairs.append((float(score_matrix[row, col]), row, col))
            pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            assigned: List[Tuple[int, int]] = []
            for _score, row, col in pairs:
                if row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                assigned.append((row, col))
            return assigned


class ECPORewardCalculator:
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
    ) -> None:
        self.model = MaxEntIRL()
        self.model.load(Path(reward_ckpt))
        self.k = int(k)
        self.gamma = float(gamma)
        self.lambda_cert = float(lambda_cert)
        self.lambda_cycle = float(lambda_cycle)
        self.invalid_penalty = float(invalid_penalty)
        self.missing_penalty = float(missing_penalty)
        self.reward_clip = reward_clip
        self.validator = ECPOValidator(k=self.k)
        self.verifier = DeterministicEvidenceVerifier(
            support_threshold=support_threshold,
            support_margin=support_margin,
        )
        self.trajectories_by_id: Dict[str, Dict[str, Any]] = {}
        if trajectory_path is not None:
            for traj in load_trajectories(Path(trajectory_path)):
                self.trajectories_by_id[_trajectory_id(traj)] = traj

    def build_window_for_meta(self, meta: Optional[Dict[str, Any]] = None) -> ECPOWindow:
        meta = meta or {}
        candidate_map = meta.get("candidate_map")
        trajectories: List[Dict[str, Any]] = []

        if isinstance(candidate_map, dict) and candidate_map:
            for candidate_id, trajectory_id in candidate_map.items():
                raw = self.trajectories_by_id.get(str(trajectory_id))
                if raw is None:
                    continue
                traj = dict(raw)
                traj["candidate_id"] = str(candidate_id)
                traj["_source_trajectory_id"] = str(trajectory_id)
                trajectories.append(traj)
        else:
            raw_candidate_ids = meta.get("candidate_ids")
            candidate_ids = raw_candidate_ids if isinstance(raw_candidate_ids, list) else []
            if candidate_ids:
                for candidate_id in candidate_ids:
                    raw = self.trajectories_by_id.get(str(candidate_id))
                    if raw is not None:
                        trajectories.append(raw)
            else:
                trajectory_id = meta.get("trajectory_id")
                if isinstance(trajectory_id, str) and trajectory_id in self.trajectories_by_id:
                    trajectories.append(self.trajectories_by_id[trajectory_id])

        if not trajectories:
            trajectories = list(self.trajectories_by_id.values())
        if not trajectories:
            raise ValueError("no trajectories available for ECPO window")

        window_id = str(meta.get("window_id") or meta.get("trajectory_id") or "window_default")
        intent_id = str(meta.get("intent_id") or "default")
        return build_window_from_trajectories(
            trajectories,
            window_id=window_id,
            intent_id=intent_id,
            skeleton_steps=meta.get("skeleton_steps"),
        )

    def score_response(self, response_text: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        window = self.build_window_for_meta(meta)
        payload, repaired = parse_policy_output(response_text)
        validation = self.validator.validate(payload, window)
        rank_reward = self._rank_reward(validation.topk, window) if payload is not None else 0.0
        cert_reward = validation.cert_reward if payload is not None else 0.0
        cycle_reward = self._cycle_reward(payload, validation, window) if payload is not None else 0.0

        invalid = 0.0 if validation.valid and not repaired else 1.0
        reward = (
            rank_reward
            + self.lambda_cert * cert_reward
            + self.lambda_cycle * cycle_reward
            - self.invalid_penalty * invalid
            - self.missing_penalty * validation.missing
        )
        if self.reward_clip is not None:
            lo, hi = self.reward_clip
            reward = float(np.clip(reward, lo, hi))

        details = {
            "rank_reward": rank_reward,
            "cert_reward": cert_reward,
            "cycle_reward": cycle_reward,
            "valid": validation.valid,
            "repaired": repaired,
            "missing": validation.missing,
            "errors": validation.errors[:5],
        }
        return float(reward), details

    def _rank_reward(self, topk: Sequence[str], window: ECPOWindow) -> float:
        if not topk:
            return 0.0
        scores = []
        for candidate_id in window.candidate_ids:
            try:
                scores.append(self.model.score(window.trajectories[candidate_id]))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("failed to score candidate %s: %s", candidate_id, exc)
                scores.append(0.0)
        arr = np.asarray(scores, dtype=np.float32)
        mean = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size and arr.std() > 1e-6 else 1.0
        score_map = {candidate_id: (score - mean) / std for candidate_id, score in zip(window.candidate_ids, scores)}
        reward = 0.0
        for index, candidate_id in enumerate(topk[: window.top_k(self.k)]):
            reward += (self.gamma**index) * float(score_map.get(candidate_id, 0.0))
        return reward / max(window.top_k(self.k), 1)

    def _cycle_reward(
        self,
        payload: Dict[str, Any],
        validation: ValidationResult,
        window: ECPOWindow,
    ) -> float:
        certificates = payload.get("certificates")
        if not isinstance(certificates, list) or validation.expected_k <= 0:
            return 0.0
        recovered = self.verifier.recover_candidates(certificates[: validation.expected_k], window)
        credit = 0
        for index in range(validation.expected_k):
            if index < len(validation.topk) and index < len(recovered) and recovered[index] == validation.topk[index]:
                credit += 1
        return credit / validation.expected_k

    def score_batch(self, metas: Sequence[Dict[str, Any]], response_texts: Sequence[str]) -> List[float]:
        rewards: List[float] = []
        for meta, response_text in zip(metas, response_texts):
            try:
                reward, details = self.score_response(response_text, meta)
                LOGGER.debug("ECPO reward details: %s", details)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("ECPO reward failed, returning zero: %s", exc)
                reward = 0.0
            rewards.append(float(reward))
        return rewards


def evaluate_certified_output(
    payload: Dict[str, Any],
    window: ECPOWindow,
    validator: ECPOValidator,
    verifier: DeterministicEvidenceVerifier,
    k: int,
) -> Dict[str, float]:
    validation = validator.validate(payload, window)
    certificates = payload.get("certificates") if isinstance(payload, dict) else []
    recovered = verifier.recover_candidates(certificates if isinstance(certificates, list) else [], window)

    topk = validation.topk[: window.top_k(k)]
    relevance = {
        candidate_id: LABEL_RELEVANCE.get(window.trajectories[candidate_id].get("label", "candidate"), 1)
        for candidate_id in window.candidate_ids
    }
    sorted_rels = sorted(relevance.values(), reverse=True)
    dcg = 0.0
    cdcg = 0.0
    idcg = 0.0
    certified_hits = 0

    for index, candidate_id in enumerate(topk, start=1):
        rel = relevance.get(candidate_id, 0)
        gain = (2**rel - 1) / np.log2(index + 1)
        dcg += gain
        is_certified = (
            index - 1 < len(recovered)
            and recovered[index - 1] == candidate_id
            and index - 1 < len(validation.bundle_results)
            and validation.bundle_results[index - 1].valid
        )
        if is_certified:
            certified_hits += 1
            cdcg += gain

    for index, rel in enumerate(sorted_rels[: window.top_k(k)], start=1):
        idcg += (2**rel - 1) / np.log2(index + 1)

    denom = window.top_k(k)
    return {
        "NDCG@K": float(dcg / idcg) if idcg > 0 else 0.0,
        "CertNDCG@K": float(cdcg / idcg) if idcg > 0 else 0.0,
        "EvidCons@K": float(certified_hits / denom) if denom else 0.0,
        "Feasible": 1.0 if validation.valid else 0.0,
    }
