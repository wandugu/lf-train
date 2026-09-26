#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 RAMS 原始 JSONL 转换为 ECPO 所需的下游文件。"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ecpo_rl.convert.common import (
    DatasetStats,
    EventArgument,
    EventConfidence,
    EventEntry,
    EventRelations,
    EventTime,
    EventTrigger,
    PreferencePair,
    RLPrompt,
    SFTSample,
    TrajectoryEntry,
    TrajectoryMeta,
    TrajectoryStep,
    build_summary,
    ensure_dir,
    write_dataset_info,
    write_jsonl,
)
from ecpo_rl.ecpo import (
    ECPO_STAGES,
    build_policy_output,
    build_window_from_trajectories,
    normalize_stage,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data" / "rams_raw"
DEFAULT_DST = ROOT / "data" / "processed"
MAPPING_DIR = ROOT / "mapping"

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("convert_rams")

AGENT_ROLE_KEYWORDS = ("agent", "attacker", "killer", "perpetrator", "suspect", "driver")
TARGET_ROLE_KEYWORDS = ("target", "victim", "place", "location")


def ensure_list(value: Optional[Iterable]) -> List:
    return list(value) if value else []


def load_mapping(path: Path, default: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if not path.exists():
        LOGGER.debug("mapping %s 不存在，使用默认值。", path)
        return default or {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        LOGGER.warning("mapping %s 格式异常，使用默认值。", path)
        return default or {}
    return payload


def flatten_sentences(sentences: List[List[str]]) -> List[str]:
    return [token for sent in sentences for token in sent]


def span_text(tokens: List[str], span: List[int]) -> str:
    start, end = span
    if start < 0 or end >= len(tokens) or start > end:
        return ""
    return " ".join(tokens[start : end + 1])


def normalise_role(role: str) -> str:
    role_lower = role.lower()
    cleaned = re.sub(r"^evt\\d+arg\\d+", "", role_lower)
    cleaned = cleaned.strip("_-") or role_lower
    return cleaned


def build_time(base_date: datetime, offset: int) -> str:
    return (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")


def load_rams_records(src_dir: Path, max_docs: Optional[int]) -> List[Dict]:
    data_dir = src_dir / "data"
    if not data_dir.exists():
        LOGGER.error("未找到 RAMS 数据目录 %s", data_dir)
        return []

    records: List[Dict] = []
    for split_name in ("train", "dev", "test"):
        path = data_dir / f"{split_name}.jsonlines"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                payload["split"] = payload.get("split", split_name)
                records.append(payload)
                if max_docs and len(records) >= max_docs:
                    LOGGER.debug("达到最大文档数 %d，停止读取。", max_docs)
                    return records
    LOGGER.info("读取 RAMS 记录 %d 条。", len(records))
    return records


def build_event_arguments(
    doc_tokens: List[str],
    trigger_span: List[int],
    links: List[List],
) -> List[EventArgument]:
    arguments: List[EventArgument] = []
    for link in links:
        if len(link) != 3:
            continue
        trigger_ref, arg_span, role = link
        if trigger_ref != trigger_span:
            continue
        role_text = normalise_role(str(role))
        entity = span_text(doc_tokens, arg_span)
        arguments.append(EventArgument(role=role_text, entity_id=entity or str(arg_span), span=arg_span))
    return arguments


def build_events(
    records: List[Dict],
    skeleton_map: Dict[str, object],
    cameo_map: Dict[str, str],
) -> List[EventEntry]:
    events: List[EventEntry] = []
    base_date = datetime(2014, 1, 1)

    for doc_idx, record in enumerate(records):
        doc_id = str(record.get("doc_key", f"doc_{doc_idx}"))
        sentences = ensure_list(record.get("sentences"))
        doc_tokens = flatten_sentences(sentences)
        doc_text = " ".join(doc_tokens)
        triggers = ensure_list(record.get("evt_triggers"))
        gold_links = ensure_list(record.get("gold_evt_links"))
        split = record.get("split")

        for trig_idx, trigger in enumerate(triggers):
            if len(trigger) < 3:
                continue
            span = trigger[0:2]
            event_type_items = ensure_list(trigger[2])
            trigger_type = event_type_items[0][0] if event_type_items else "unknown"
            trigger_text = span_text(doc_tokens, span)
            arguments = build_event_arguments(doc_tokens, span, gold_links)
            time_value = build_time(base_date, doc_idx * 2 + trig_idx)
            skeleton_hits = resolve_skeleton_hits(trigger_type, skeleton_map)
            skeleton_type = skeleton_hits[0] if skeleton_hits else "UNKNOWN"

            mapping = {
                "cameo": cameo_map.get(trigger_type, "000"),
                "skeleton_type": skeleton_type,
            }

            try:
                entry = EventEntry(
                    doc_id=doc_id,
                    event_id=f"{doc_id}_event_{trig_idx}",
                    trigger=EventTrigger(span=span, text=trigger_text, type=trigger_type),
                    arguments=arguments,
                    time=EventTime(value=time_value, span=span),
                    relations=EventRelations(temporal=[], causal=[], subevent=[]),
                    confidence=EventConfidence(trigger_prob=1.0, arg_role_avg=1.0),
                    source="RAMS",
                    mapping=mapping,
                    split=split,
                )
                entry.mapping["doc_text"] = doc_text[:120]
            except ValidationError as exc:
                LOGGER.error("事件 %s 校验失败：%s", doc_id, exc)
                continue
            events.append(entry)
    LOGGER.info("生成 RAMS 事件 %d 条。", len(events))
    return events


def resolve_skeleton_hits(event_type: str, skeleton_map: Dict[str, object]) -> List[str]:
    mapping = skeleton_map.get(event_type)
    if isinstance(mapping, list):
        return [str(item) for item in mapping if item]
    if isinstance(mapping, str):
        return [mapping]
    return ["PREP"]


def compute_delta_days(prev_time: Optional[str], current_time: str) -> int:
    if not prev_time:
        return 0
    prev = datetime.strptime(prev_time, "%Y-%m-%d")
    curr = datetime.strptime(current_time, "%Y-%m-%d")
    return max((curr - prev).days, 0)


def infer_agent_ids(arguments: List[EventArgument], fallback: str) -> List[str]:
    agent_ids: List[str] = []
    for arg in arguments:
        role_lower = arg.role.lower()
        if any(keyword in role_lower for keyword in AGENT_ROLE_KEYWORDS):
            agent_ids.append(arg.entity_id)
    return agent_ids or [fallback]


def classify_label(skeleton_seq: List[str]) -> str:
    if "EXECUTE" in skeleton_seq:
        return "expert"
    if "PREP" in skeleton_seq:
        return "candidate"
    return "negative"


def build_trajectories(events: List[EventEntry], skeleton_map: Dict[str, object]) -> List[TrajectoryEntry]:
    grouped: Dict[str, List[EventEntry]] = defaultdict(list)
    for event in events:
        agents = infer_agent_ids(event.arguments, f"doc::{event.doc_id}")
        for agent in agents:
            grouped[agent].append(event)

    trajectories: List[TrajectoryEntry] = []
    for person_id, ev_list in grouped.items():
        ev_list.sort(key=lambda e: e.time.value)
        steps: List[TrajectoryStep] = []
        meta_nodes = {person_id}
        meta_edges: List[List[str]] = []
        skeleton_seq: List[str] = []
        prev_time: Optional[str] = None

        for event in ev_list:
            roles = {arg.role: arg.entity_id for arg in event.arguments}
            meta_nodes.update(roles.values())
            skeleton_hits = resolve_skeleton_hits(event.trigger.type, skeleton_map)
            skeleton_seq.extend(skeleton_hits)
            delta = compute_delta_days(prev_time, event.time.value)
            prev_time = event.time.value
            steps.append(
                TrajectoryStep(
                    event_id=event.event_id,
                    time=event.time.value,
                    type=event.trigger.type or "Unknown",
                    roles=roles,
                    delta_days_from_prev=delta,
                    text_refs=[{"doc_id": event.doc_id, "span": event.trigger.span}],
                    skeleton_hits=skeleton_hits,
                )
            )
            for role, value in roles.items():
                if any(keyword in role.lower() for keyword in TARGET_ROLE_KEYWORDS):
                    meta_edges.append([person_id, event.trigger.type, value, event.time.value])

        if not steps:
            continue
        traj_id = f"traj_{person_id}"
        label = classify_label(skeleton_seq)
        trajectories.append(
            TrajectoryEntry(
                person_id=person_id,
                trajectory_id=traj_id,
                label=label,
                steps=steps,
                meta=TrajectoryMeta(graph_nodes=sorted(meta_nodes), graph_edges=meta_edges),
            )
        )

    LOGGER.info("生成 RAMS 轨迹 %d 条。", len(trajectories))
    return trajectories


def degrade_trajectory(traj: TrajectoryEntry) -> Optional[TrajectoryEntry]:
    if len(traj.steps) < 2:
        return None
    steps = [step.model_copy(deep=True) for step in traj.steps[:-1]]
    degraded = TrajectoryEntry(
        person_id=traj.person_id,
        trajectory_id=f"{traj.trajectory_id}_neg",
        label="negative",
        steps=steps,
        meta=traj.meta,
    )
    return degraded


def build_preference_pairs(
    trajectories: List[TrajectoryEntry],
) -> Tuple[List[TrajectoryEntry], List[PreferencePair]]:
    extra_trajs: List[TrajectoryEntry] = []
    pairs: List[PreferencePair] = []
    for traj in trajectories:
        degraded = degrade_trajectory(traj)
        if not degraded:
            continue
        extra_trajs.append(degraded)
        pairs.append(
            PreferencePair(
                better=traj.trajectory_id,
                worse=degraded.trajectory_id,
                reason="原始轨迹包含更多事件步骤，可信度更高。",
            )
        )
    LOGGER.info("生成偏好对 %d 条。", len(pairs))
    return extra_trajs, pairs


def build_sft_samples(events: List[EventEntry]) -> Tuple[List[SFTSample], Dict[str, List[SFTSample]]]:
    samples: List[SFTSample] = []
    split_map: Dict[str, List[SFTSample]] = defaultdict(list)

    for event in events:
        arguments = ", ".join(f"{arg.role}:{arg.entity_id}" for arg in event.arguments)
        sample = SFTSample(
            instruction="请识别事件类型并抽取关键论元。",
            input=f"触发词：{event.trigger.text}\n事件上下文：{event.mapping.get('doc_text', '')}",
            output=f"事件类型：{event.trigger.type}\n论元：{arguments or '无'}",
            system="你是事件抽取助手。",
        )
        samples.append(sample)
        split = event.split or "train"
        split_map[split].append(sample)
    LOGGER.info("生成 SFT 样本 %d 条。", len(samples))
    return samples, split_map


def _format_ecpo_window_prompt(window) -> str:
    lines = [
        f"[WINDOW] {window.window_id}",
        f"[INTENT] {window.intent_id}",
        "[SKELETON] " + " -> ".join(window.skeleton_steps),
        "[CANDIDATES]",
    ]
    for candidate_id in window.candidate_ids:
        traj = window.trajectories[candidate_id]
        lines.append(f"- candidate_id={candidate_id}")
        for step in traj.get("steps", []):
            roles_str = ", ".join(f"{k}:{v}" for k, v in step.get("roles", {}).items())
            skeleton_str = ",".join(normalize_stage(item) for item in step.get("skeleton_hits", [])) or "PREP"
            refs = ";".join(f"{ref.get('doc_id')}:{ref.get('span')}" for ref in step.get("text_refs", []))
            lines.append(
                f"  * event_id={step.get('event_id')} time={step.get('time')} type={step.get('type')} "
                f"roles={{{roles_str}}} skeleton={skeleton_str} evidence={refs}"
            )
    lines.extend(
        [
            "[OUTPUT]",
            "Return strict JSON only with keys: window_id, topk, certificates.",
            "topk must contain window-local candidate_id values only.",
            "certificates must align by rank position and include one step per skeleton stage.",
            "Each matched step cites doc_id/span evidence; unmatched steps use matched=false, event_id=null, evidence=[].",
        ]
    )
    return "\n".join(lines)


def _reference_rank(candidate_ids: List[str], trajectories: Dict[str, Dict[str, object]]) -> List[str]:
    label_rank = {"expert": 2, "candidate": 1, "negative": 0}
    return sorted(
        candidate_ids,
        key=lambda cid: (
            -label_rank.get(str(trajectories[cid].get("label", "candidate")), 1),
            -len(trajectories[cid].get("steps", [])),
            cid,
        ),
    )


def build_rl_prompts(trajectories: List[TrajectoryEntry]) -> List[RLPrompt]:
    prompts: List[RLPrompt] = []
    rng = random.Random(42)
    sorted_trajs = sorted(trajectories, key=lambda item: item.trajectory_id)
    window_size = 10
    for window_idx, start in enumerate(range(0, len(sorted_trajs), window_size), start=1):
        chunk = sorted_trajs[start : start + window_size]
        if not chunk:
            continue
        chunk = list(chunk)
        rng.shuffle(chunk)
        localized: List[Dict[str, object]] = []
        candidate_map: Dict[str, str] = {}
        for local_idx, traj in enumerate(chunk, start=1):
            candidate_id = f"C{local_idx:03d}"
            payload = traj.model_dump()
            payload["candidate_id"] = candidate_id
            localized.append(payload)
            candidate_map[candidate_id] = traj.trajectory_id

        window_id = f"rams_window_{window_idx:05d}"
        window = build_window_from_trajectories(
            localized,
            window_id=window_id,
            intent_id="RAMS",
            skeleton_steps=ECPO_STAGES,
        )
        ranked_ids = _reference_rank(window.candidate_ids, window.trajectories)
        response = json.dumps(build_policy_output(window, ranked_ids, k=10), ensure_ascii=False)
        top_candidate = ranked_ids[0] if ranked_ids else window.candidate_ids[0]
        source_traj_id = candidate_map.get(top_candidate, chunk[0].trajectory_id)
        prompts.append(
            RLPrompt(
                prompt=_format_ecpo_window_prompt(window),
                response=response,
                trajectory_id=source_traj_id,
                person_id=chunk[0].person_id,
                window_id=window.window_id,
                intent_id=window.intent_id,
                candidate_ids=window.candidate_ids,
                candidate_map=candidate_map,
                skeleton_steps=window.skeleton_steps,
            )
        )
    LOGGER.info("构造 ECPO RL 窗口提示 %d 条。", len(prompts))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RAMS raw JSONL to processed ECPO datasets")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="原始 RAMS 目录")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="输出目录")
    parser.add_argument("--max-docs", type=int, default=None, help="最多处理的文档数")
    args = parser.parse_args()

    ensure_dir(args.dst)

    skeleton_map = load_mapping(MAPPING_DIR / "event2skeleton.json", default={})
    cameo_map = load_mapping(MAPPING_DIR / "event2cameo.json", default={})

    records = load_rams_records(args.src, args.max_docs)
    if not records:
        LOGGER.error("未读取到 RAMS 原始数据。")
        return

    events = build_events(records, skeleton_map, cameo_map)
    if not events:
        LOGGER.error("未生成任何事件，流程终止。")
        return

    event_stats = write_jsonl(args.dst / "rams_event.jsonl", events)

    trajectories = build_trajectories(events, skeleton_map)
    extra_trajs, pairs = build_preference_pairs(trajectories)
    all_trajs = trajectories + extra_trajs
    traj_stats = write_jsonl(args.dst / "rams_traj.jsonl", all_trajs)
    pair_stats = write_jsonl(args.dst / "rams_pairs.jsonl", pairs)

    sft_samples, sft_by_split = build_sft_samples(events)
    sft_stats = write_jsonl(args.dst / "rams_sft.jsonl", sft_samples)
    sft_split_stats: List[DatasetStats] = []
    for split_name, split_samples in sorted(sft_by_split.items()):
        split_path = args.dst / f"rams_sft_{split_name}.jsonl"
        sft_split_stats.append(write_jsonl(split_path, split_samples))

    rl_prompts = build_rl_prompts(all_trajs)
    rl_stats = write_jsonl(args.dst / "rams_rl_prompts.jsonl", rl_prompts)

    write_dataset_info(
        args.dst,
        {
            "rams_sft": {
                "file_name": "rams_sft.jsonl",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history",
                },
            },
            "rams_rl": {
                "file_name": "rams_rl_prompts.jsonl",
                "formatting": "ecpo_rl",
                "columns": {"prompt": "prompt", "response": "response"},
            },
        },
    )

    summary_path = args.dst / "rams_summary.stats.json"
    summary_items = [event_stats, traj_stats, pair_stats, sft_stats]
    summary_items.extend(sft_split_stats)
    summary_items.append(rl_stats)
    build_summary(summary_items, summary_path)
    LOGGER.info("处理完成，结果写入 %s", args.dst)


if __name__ == "__main__":
    main()
