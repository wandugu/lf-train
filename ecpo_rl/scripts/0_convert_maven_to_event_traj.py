#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 MAVEN 系列原始 JSON 转换为 ECPO 所需的五类下游文件。

脚本逻辑概览：
1. 读取 ``data/maven_raw`` 下的 JSON 文档或 ``train/valid/test.jsonl`` 拆分，并结合映射表构造 ``event.jsonl``。
2. 基于 Agent 论元汇聚事件，生成按月划分的 person 轨迹 ``traj.jsonl``。
3. 对每条轨迹构造降质版本，写出偏好对 ``pairs.jsonl``。
4. 将事件转写成指令微调样本 ``maven_sft.jsonl`` 及按拆分输出的 ``maven_sft_{split}.jsonl``。
5. 将轨迹截断为前缀提示生成 ``rl_prompts.jsonl``。

所有输出均通过 Pydantic schema 校验，同时生成统计信息 ``*.stats.json`` 与汇总 ``summary.stats.json``。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pydantic import ValidationError
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ecpo_rl.convert.common import (
    DatasetStats,
    EventArgument,
    EventConfidence,
    EventEntry,
    EventRelation,
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
DEFAULT_SRC = ROOT / "data" / "maven_raw"
DEFAULT_DST = ROOT / "data" / "processed"
MAPPING_DIR = ROOT / "mapping"

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("convert_maven")


# =============================
# 解析原始 JSON
# =============================


def load_mapping(path: Path, default: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    LOGGER.warning("未找到映射文件 %s，使用默认值。", path)
    return default or {}


def normalise_span(value: Dict[str, int] | List[int]) -> List[int]:
    if isinstance(value, list):
        assert len(value) == 2
        return [int(value[0]), int(value[1])]
    keys = ["start", "end"]
    if all(k in value for k in keys):
        return [int(value["start"]), int(value["end"])]
    if all(k in value for k in ("start_token", "end_token")):
        return [int(value["start_token"]), int(value["end_token"])]
    if "offset" in value and isinstance(value["offset"], list) and len(value["offset"]) == 2:
        return [int(value["offset"][0]), int(value["offset"][1])]
    raise ValueError(f"无法解析 span: {value}")


def ensure_time(event: Dict[str, object], fallback_date: str) -> Tuple[str, List[int]]:
    raw_time = event.get("time")
    if isinstance(raw_time, dict) and "value" in raw_time:
        try:
            datetime.strptime(raw_time["value"], "%Y-%m-%d")
            return raw_time["value"], normalise_span(raw_time.get("span", [0, 0]))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("时间字段解析失败 %s：%s，使用回退日期 %s", event.get("id"), exc, fallback_date)
    return fallback_date, [0, 0]


def extract_trigger_from_mentions(
    event: Dict[str, object],
    doc_content: Optional[List[Dict[str, object]]],
) -> Tuple[str, List[int]]:
    mentions = event.get("mention") or event.get("mentions") or []
    if isinstance(mentions, dict):
        mentions = [mentions]
    if not mentions:
        return "", [0, 0]
    mention = mentions[0]
    trigger_text = mention.get("trigger_word") or mention.get("text") or ""
    offset = mention.get("offset") or mention.get("span") or [0, 0]
    if not trigger_text and doc_content:
        sent_id = mention.get("sent_id")
        if isinstance(sent_id, int) and 0 <= sent_id < len(doc_content):
            tokens = doc_content[sent_id].get("tokens", [])
            if (
                isinstance(tokens, list)
                and isinstance(offset, list)
                and len(offset) == 2
            ):
                start, end = offset
                start = max(int(start), 0)
                end = max(int(end), start)
                slice_tokens = tokens[start:end]
                if slice_tokens and all(isinstance(tok, str) for tok in slice_tokens):
                    trigger_text = " ".join(slice_tokens).strip()
    try:
        span = normalise_span(offset)
    except Exception:  # noqa: BLE001
        span = [0, 0]
    return trigger_text, span


EVENT_TYPE_CANDIDATE_KEYS = (
    "type",
    "event_type",
    "event_subtype",
    "subtype",
    "label",
)


def resolve_event_type(event: Dict[str, object]) -> str:
    for key in EVENT_TYPE_CANDIDATE_KEYS:
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    type_id = event.get("type_id")
    if isinstance(type_id, str) and type_id.strip():
        return type_id.strip()
    if isinstance(type_id, int):
        return str(type_id)
    mentions = event.get("mention") or event.get("mentions") or []
    if isinstance(mentions, dict):
        mentions = [mentions]
    for mention in mentions:
        value = mention.get("type")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown"


def parse_event(
    doc_id: str,
    event: Dict[str, object],
    skeleton_map: Dict[str, object],
    cameo_map: Dict[str, str],
    fallback_date: str,
    doc_content: Optional[List[Dict[str, object]]] = None,
    split: Optional[str] = None,
) -> Optional[EventEntry]:
    trigger = event.get("trigger") or {}
    trigger_type = resolve_event_type(event)
    using_new_schema = "trigger" not in event or not trigger
    is_candidate = bool(event.get("_candidate_event"))

    if using_new_schema:
        trigger_text, trigger_span = extract_trigger_from_mentions(event, doc_content)
        if not trigger_text:
            LOGGER.warning("事件 %s 缺少触发词，已跳过", event.get("id"))
            return None
        time_value, time_span = fallback_date, [0, 0]
        arguments: List[EventArgument] = []
        relations_dict: Dict[str, List[Dict[str, str]]] = {}
        source = "MAVEN-CANDIDATE" if is_candidate else "MAVEN-JSONL"
    else:
        trigger_text = trigger.get("text", "")
        if not trigger_text:
            LOGGER.warning("事件 %s 缺少 trigger.text，已跳过", event.get("id"))
            return None
        try:
            trigger_span = normalise_span(trigger)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("事件 %s 的 trigger span 解析失败：%s", event.get("id"), exc)
            trigger_span = [0, 0]
        time_value, time_span = ensure_time(event, fallback_date)

        arguments = []
        for arg in event.get("arguments", []):
            role = arg.get("role", "")
            if not role:
                continue
            entity_id = arg.get("entity_id") or arg.get("text") or f"{trigger_type}_{role}"
            try:
                span = normalise_span(arg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("事件 %s 的 argument span 解析失败：%s", event.get("id"), exc)
                span = [0, 0]
            arguments.append(EventArgument(role=role, entity_id=str(entity_id), span=span))

        if not arguments:
            LOGGER.warning("事件 %s 不含 argument，已跳过", event.get("id"))
            return None

        relations_dict = event.get("relations", {})
        source = "MAVEN|MAVEN-Arg|MAVEN-ERE|RAMS"

    relations = EventRelations(
        temporal=[EventRelation(**rel) for rel in relations_dict.get("temporal", [])],
        causal=[EventRelation(**rel) for rel in relations_dict.get("causal", [])],
        subevent=[EventRelation(**rel) for rel in relations_dict.get("subevent", [])],
    )

    skeleton_hits = resolve_skeleton_hits(trigger_type, skeleton_map)
    skeleton_type = skeleton_hits[0] if skeleton_hits else "UNKNOWN"
    mapping = {
        "cameo": cameo_map.get(trigger_type, "000"),
        "skeleton_type": skeleton_type,
    }

    confidence = EventConfidence(trigger_prob=0.9, arg_role_avg=0.85)

    try:
        entry = EventEntry(
            doc_id=doc_id,
            event_id=event.get("id", f"{doc_id}_event"),
            trigger=EventTrigger(span=trigger_span, text=trigger_text, type=trigger_type),
            arguments=arguments,
            time=EventTime(value=time_value, span=time_span),
            relations=relations,
            confidence=confidence,
            source=source,
            mapping=mapping,
            split=split,
        )
    except ValidationError as exc:
        LOGGER.error("事件 %s 校验失败：%s", event.get("id"), exc)
        return None
    return entry


def load_events(src_dir: Path, skeleton_map: Dict[str, object], cameo_map: Dict[str, str]) -> List[EventEntry]:
    events: List[EventEntry] = []
    if not src_dir.exists():
        LOGGER.error("原始目录 %s 不存在。", src_dir)
        return events

    split_files = {name: src_dir / f"{name}.jsonl" for name in ("train", "valid", "test")}
    has_jsonl = any(path.exists() for path in split_files.values())

    if has_jsonl:
        for split_name, jsonl_path in split_files.items():
            if not jsonl_path.exists():
                continue
            with jsonl_path.open("r", encoding="utf-8") as f:
                iterator = enumerate(tqdm(f, desc=f"加载{split_name}", unit="doc"), start=1)
                for line_idx, line in iterator:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:  # noqa: PERF203
                        LOGGER.error("解析 %s 第 %d 行失败：%s", jsonl_path.name, line_idx, exc)
                        continue
                    doc_id = data.get("id") or f"{split_name}_{line_idx:06d}"
                    fallback_date = (
                        data.get("publish_time")
                        or data.get("time")
                        or data.get("date")
                        or "2014-01-01"
                    )
                    doc_content = data.get("content")
                    raw_events = data.get("events") or []
                    if not raw_events and data.get("candidates"):
                        raw_events = [
                            {
                                "id": cand.get("id"),
                                "type": "Unknown",
                                "mention": [cand],
                                "_candidate_event": True,
                            }
                            for cand in data.get("candidates", [])
                        ]
                    for event in raw_events:
                        entry = parse_event(
                            doc_id,
                            event,
                            skeleton_map,
                            cameo_map,
                            fallback_date,
                            doc_content=doc_content,
                            split=split_name,
                        )
                        if entry is not None:
                            events.append(entry)
        LOGGER.info(
            "共解析事件 %d 条，来源拆分：%s。",
            len(events),
            ", ".join(name for name, path in split_files.items() if path.exists()),
        )
        return events

    json_files = sorted(path for path in src_dir.glob("*.json"))
    if not json_files:
        LOGGER.error("原始目录 %s 下未找到 JSON/JSONL 文件。", src_dir)
        return events

    for json_path in tqdm(json_files, desc="加载文档"):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data.get("id") or json_path.stem
        fallback_date = data.get("publish_time") or "2014-01-01"
        raw_events = data.get("events") or []
        if not raw_events and data.get("candidates"):
            raw_events = [
                {
                    "id": cand.get("id"),
                    "type": "Unknown",
                    "mention": [cand],
                    "_candidate_event": True,
                }
                for cand in data.get("candidates", [])
            ]
        for event in raw_events:
            entry = parse_event(
                doc_id,
                event,
                skeleton_map,
                cameo_map,
                fallback_date,
            )
            if entry is not None:
                events.append(entry)
    LOGGER.info("共解析事件 %d 条。", len(events))
    return events


# =============================
# 构造轨迹与偏好对
# =============================


def month_key(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m")


def compute_delta_days(prev: Optional[str], current: str) -> int:
    if prev is None:
        return 0
    dt_prev = datetime.strptime(prev, "%Y-%m-%d")
    dt_cur = datetime.strptime(current, "%Y-%m-%d")
    return (dt_cur - dt_prev).days


def classify_label(skeleton_seq: List[str]) -> str:
    hits = set(skeleton_seq)
    if {"PREP", "PROBE", "EXECUTE", "OUTCOME"}.issubset(hits):
        return "expert"
    if len(hits) >= 2 and "EXECUTE" in hits:
        return "candidate"
    return "negative"


AGENT_ROLE_KEYWORDS = {
    "agent",
    "perpetrator",
    "actor",
    "attacker",
    "suspect",
    "subject",
    "person",
}


SKELETON_STAGE_ORDER = ["PREP", "PROBE", "EXECUTE", "OUTCOME"]
PREP_KEYWORDS = {
    "plan",
    "prepare",
    "gather",
    "meet",
    "statement",
    "warn",
    "warning",
    "know",
    "intel",
    "intelligence",
    "discuss",
    "train",
    "process_start",
    "mobilize",
    "mobilise",
}
PROBE_KEYWORDS = {
    "probe",
    "observe",
    "monitor",
    "investigate",
    "analyse",
    "analyze",
    "assess",
    "inspect",
    "review",
    "survey",
    "recon",
    "scout",
    "interrogate",
    "warning",
    "statement",
    "process_start",
    "negotiat",
}
EXECUTE_KEYWORDS = {
    "attack",
    "hostile",
    "strike",
    "bomb",
    "kill",
    "killing",
    "kidnap",
    "shoot",
    "raid",
    "explode",
    "explosion",
    "execute",
    "combat",
    "military",
    "operation",
    "detain",
    "arrest",
    "arriv",
    "threat",
    "destroy",
    "damage",
    "riot",
    "protest",
    "conflict",
    "catastrophe",
    "violence",
}
OUTCOME_KEYWORDS = {
    "loot",
    "cashout",
    "withdraw",
    "escape",
    "ransom",
    "benefit",
    "profit",
    "gain",
    "sell",
    "launder",
    "transport",
    "transfer",
    "process_end",
    "aftermath",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "has",
    "have",
    "had",
    "that",
    "this",
    "these",
    "those",
}


def infer_skeleton_from_type(event_type: str) -> List[str]:
    event_type_lower = (event_type or "").lower()
    hits: List[str] = []
    if any(keyword in event_type_lower for keyword in EXECUTE_KEYWORDS):
        hits.append("EXECUTE")
    if any(keyword in event_type_lower for keyword in OUTCOME_KEYWORDS):
        hits.append("OUTCOME")
    if any(keyword in event_type_lower for keyword in PROBE_KEYWORDS):
        hits.insert(0, "PROBE")
    if any(keyword in event_type_lower for keyword in PREP_KEYWORDS):
        if "PROBE" not in hits:
            hits.insert(0, "PROBE")
        if "PREP" not in hits:
            hits.insert(0, "PREP")
    if not hits:
        hits = ["PREP"]
    ordered_hits: List[str] = []
    for stage in SKELETON_STAGE_ORDER:
        if stage in hits and stage not in ordered_hits:
            ordered_hits.append(stage)
    return ordered_hits or ["PREP"]


def resolve_skeleton_hits(event_type: str, skeleton_map: Dict[str, object]) -> List[str]:
    raw_hits = skeleton_map.get(event_type)
    hits: List[str] = []
    if isinstance(raw_hits, str) and raw_hits.strip():
        hits = [raw_hits.strip().upper()]
    elif isinstance(raw_hits, list):
        for item in raw_hits:
            if isinstance(item, str) and item.strip():
                hits.append(item.strip().upper())
    if hits:
        ordered: List[str] = []
        for stage in SKELETON_STAGE_ORDER:
            if stage in hits and stage not in ordered:
                ordered.append(stage)
        return ordered or hits
    return infer_skeleton_from_type(event_type)


def _merge_titlecase_tokens(tokens: List[str]) -> List[str]:
    merged: List[str] = []
    buffer: List[str] = []
    for tok in tokens:
        clean = tok.strip(".,;:'\"()[]{}")
        if not clean:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            continue
        first = clean[0]
        if first.isupper() and clean.lower() not in STOPWORDS:
            buffer.append(clean)
        else:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
    if buffer:
        merged.append(" ".join(buffer))
    return merged


def infer_core_arguments(event: EventEntry, doc_ctx: Optional[MavenDocContext]) -> Dict[str, str]:
    if doc_ctx is None:
        return {"Trigger": event.trigger.text or event.trigger.type or "UNKNOWN"}
    sent_id = resolve_sent_id(event, doc_ctx)
    sentence = ""
    tokens: List[str] = []
    if 0 <= sent_id < len(doc_ctx.sentences):
        sentence = doc_ctx.sentences[sent_id]
    if 0 <= sent_id < len(doc_ctx.tokens):
        tokens = doc_ctx.tokens[sent_id]
    if not tokens and sentence:
        tokens = [tok for tok in sentence.split() if tok]
    candidates = _merge_titlecase_tokens(tokens)
    arguments: Dict[str, str] = {}
    if candidates:
        arguments["Agent"] = candidates[0]
        if len(candidates) > 1:
            arguments["Target"] = candidates[1]
    if not arguments and sentence:
        snippet = sentence.strip()
        arguments["Context"] = snippet[:120]
    if not arguments:
        arguments["Trigger"] = event.trigger.text or event.trigger.type or "UNKNOWN"
    return arguments


def _collect_agent_candidates(event: EventEntry) -> List[str]:
    candidates: List[str] = []
    for arg in event.arguments:
        role = (arg.role or "").strip().lower()
        if any(keyword in role for keyword in AGENT_ROLE_KEYWORDS):
            candidates.append(arg.entity_id)
    return candidates


def build_trajectories(
    events: List[EventEntry],
    skeleton_map: Dict[str, object],
    context_index: Optional[Dict[str, MavenDocContext]] = None,
) -> Tuple[List[TrajectoryEntry], Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Tuple[EventEntry, str]]] = defaultdict(list)
    for event in events:
        agent_ids = _collect_agent_candidates(event)
        if not agent_ids:
            fallback_agent = f"doc::{event.doc_id}"
            agent_ids = [fallback_agent]
        for agent in agent_ids:
            grouped[(agent, month_key(event.time.value))].append((event, agent))

    trajectories: List[TrajectoryEntry] = []
    base_to_person: Dict[str, str] = {}

    for (person_id, month), ev_list in grouped.items():
        ev_list.sort(key=lambda item: datetime.strptime(item[0].time.value, "%Y-%m-%d"))
        steps: List[TrajectoryStep] = []
        meta_nodes = set([person_id])
        meta_edges: List[List[str]] = []
        prev_time: Optional[str] = None
        skeleton_seq: List[str] = []
        for event, _agent in ev_list:
            doc_ctx = context_index.get(event.doc_id) if context_index else None
            real_roles = {arg.role: arg.entity_id for arg in event.arguments}
            roles = real_roles or infer_core_arguments(event, doc_ctx)
            if real_roles:
                meta_nodes.update(real_roles.values())
            else:
                for value in roles.values():
                    if isinstance(value, str) and len(value) > 80:
                        meta_nodes.add(value[:80] + '…')
                    else:
                        meta_nodes.add(value)
            skeleton_hits = resolve_skeleton_hits(event.trigger.type, skeleton_map)
            skeleton_seq.extend(skeleton_hits)
            delta = compute_delta_days(prev_time, event.time.value)
            prev_time = event.time.value
            text_ref = {"doc_id": event.doc_id, "span": event.trigger.span}
            step_type = event.trigger.type or "Unknown"
            steps.append(
                TrajectoryStep(
                    event_id=event.event_id,
                    time=event.time.value,
                    type=step_type,
                    roles=roles,
                    delta_days_from_prev=delta,
                    text_refs=[text_ref],
                    skeleton_hits=skeleton_hits,
                )
            )
            for role, ent in roles.items():
                role_lower = role.lower()
                if role_lower in {"target", "place", "location"}:
                    meta_edges.append([person_id, event.trigger.type, ent, event.time.value])
        if not steps:
            continue
        label = classify_label(skeleton_seq)
        traj_id = f"traj_{person_id}_{month}"
        trajectory = TrajectoryEntry(
            person_id=person_id,
            trajectory_id=traj_id,
            label=label,
            steps=steps,
            meta=TrajectoryMeta(graph_nodes=sorted(meta_nodes), graph_edges=meta_edges),
        )
        trajectories.append(trajectory)
        base_to_person[traj_id] = person_id

    LOGGER.info("生成轨迹 %d 条。", len(trajectories))
    return trajectories, base_to_person


def degrade_trajectory(traj: TrajectoryEntry, suffix: str, rng: random.Random) -> Optional[TrajectoryEntry]:
    steps = list(traj.steps)
    if len(steps) < 2:
        return None
    mutated_steps = [step.model_copy(deep=True) for step in steps]
    choice = rng.choice(["drop", "swap"])
    if choice == "drop":
        drop_count = 1 if len(mutated_steps) <= 3 else 2
        drop_indices = sorted(rng.sample(range(len(mutated_steps)), drop_count), reverse=True)
        for idx in drop_indices:
            mutated_steps.pop(idx)
    else:  # swap
        idx = rng.randrange(len(mutated_steps) - 1)
        mutated_steps[idx], mutated_steps[idx + 1] = mutated_steps[idx + 1], mutated_steps[idx]
    if not mutated_steps:
        return None
    mutated_label = "candidate" if choice == "swap" else "negative"
    mutated_id = f"{traj.trajectory_id}_{suffix}"
    mutated = TrajectoryEntry(
        person_id=traj.person_id,
        trajectory_id=mutated_id,
        label=mutated_label,
        steps=mutated_steps,
        meta=traj.meta,
    )
    return mutated


def build_preference_pairs(
    trajectories: List[TrajectoryEntry],
    rng: random.Random,
) -> Tuple[List[TrajectoryEntry], List[PreferencePair]]:
    extra_trajs: List[TrajectoryEntry] = []
    pairs: List[PreferencePair] = []
    for traj in trajectories:
        mutated = degrade_trajectory(traj, "mut", rng)
        if mutated is None:
            continue
        extra_trajs.append(mutated)
        pairs.append(
            PreferencePair(
                better=traj.trajectory_id,
                worse=mutated.trajectory_id,
                reason="more complete EXECUTE->OUTCOME chain",
            )
        )
    LOGGER.info("生成偏好对 %d 组。", len(pairs))
    return extra_trajs, pairs


# =============================
# 构造 SFT 与 RL 数据
# =============================


@dataclass
class MavenDocContext:
    sentences: List[str] = field(default_factory=list)
    tokens: List[List[str]] = field(default_factory=list)
    event_sent_ids: Dict[str, Optional[int]] = field(default_factory=dict)
    triggers_by_text: Dict[str, List[Optional[int]]] = field(default_factory=dict)


def _extract_sentence_text(entry: object) -> Tuple[str, List[str]]:
    if isinstance(entry, dict):
        sentence = entry.get("sentence") if isinstance(entry.get("sentence"), str) else None
        if sentence:
            text = sentence.strip()
            tokens = [tok for tok in text.split() if tok]
            return text, tokens
        tokens = entry.get("tokens")
        if isinstance(tokens, list):
            token_strs = [tok for tok in tokens if isinstance(tok, str)]
            if token_strs:
                text = " ".join(token_strs).strip()
                clean_tokens = [tok for tok in token_strs if tok]
                return text, clean_tokens
    elif isinstance(entry, str):
        text = entry.strip()
        tokens = [tok for tok in text.split() if tok]
        return text, tokens
    return "", []


def _iter_source_documents(src_dir: Path) -> Iterable[Tuple[str, Dict[str, object]]]:
    split_files = {name: src_dir / f"{name}.jsonl" for name in ("train", "valid", "test")}
    has_jsonl = any(path.exists() for path in split_files.values())
    if has_jsonl:
        for split_name, jsonl_path in split_files.items():
            if not jsonl_path.exists():
                continue
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:  # noqa: PERF203
                        LOGGER.warning("跳过 %s 第 %d 行：%s", jsonl_path.name, line_idx, exc)
                        continue
                    doc_id = data.get("id") or f"{split_name}_{line_idx:06d}"
                    yield doc_id, data
    else:
        for json_path in sorted(src_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:  # noqa: PERF203
                LOGGER.warning("解析 %s 失败：%s", json_path.name, exc)
                continue
            doc_id = data.get("id") or json_path.stem
            yield doc_id, data


def build_doc_context_index(src_dir: Path, target_doc_ids: Set[str]) -> Dict[str, MavenDocContext]:
    contexts: Dict[str, MavenDocContext] = {}
    if not target_doc_ids:
        return contexts
    remaining = set(target_doc_ids)
    for doc_id, data in _iter_source_documents(src_dir):
        if doc_id not in remaining:
            continue
        content = data.get("content") or []
        sentences: List[str] = []
        sentence_tokens: List[List[str]] = []
        if isinstance(content, list):
            for entry in content:
                sent_text, sent_tokens = _extract_sentence_text(entry)
                if not sent_text and sent_tokens:
                    sent_text = " ".join(sent_tokens)
                if sent_text:
                    sentences.append(sent_text)
                    if sent_tokens:
                        sentence_tokens.append(sent_tokens)
                    else:
                        sentence_tokens.append([tok for tok in sent_text.split() if tok])
        event_sent_ids: Dict[str, Optional[int]] = {}
        triggers_by_text: Dict[str, List[Optional[int]]] = defaultdict(list)
        events = data.get("events") or []
        if not events and data.get("candidates"):
            events = data.get("candidates", [])
        for idx, raw_event in enumerate(events):
            event_id = raw_event.get("id") or f"{doc_id}_event"
            mentions = raw_event.get("mention") or raw_event.get("mentions") or []
            if isinstance(mentions, dict):
                mentions = [mentions]
            sent_id: Optional[int] = None
            trigger_word: Optional[str] = None
            for mention in mentions:
                if trigger_word is None:
                    trigger_word = (
                        mention.get("trigger_word")
                        or mention.get("text")
                        or mention.get("trigger")
                    )
                candidate = mention.get("sent_id")
                if isinstance(candidate, int):
                    sent_id = candidate
                    break
            trigger = raw_event.get("trigger")
            if trigger_word is None and isinstance(trigger, dict):
                trigger_word = trigger.get("text")
            if isinstance(trigger, dict) and sent_id is None:
                candidate = trigger.get("sent_id")
                if isinstance(candidate, int):
                    sent_id = candidate
            trigger_word = (trigger_word or "").strip()
            event_sent_ids[event_id] = sent_id
            if trigger_word:
                triggers_by_text[trigger_word.lower()].append(sent_id)
        contexts[doc_id] = MavenDocContext(
            sentences=sentences,
            tokens=sentence_tokens,
            event_sent_ids=event_sent_ids,
            triggers_by_text={k: list(v) for k, v in triggers_by_text.items()},
        )
        remaining.discard(doc_id)
        if not remaining:
            break
    if remaining:
        LOGGER.warning("部分文档缺少上下文信息：%s", ", ".join(sorted(remaining)[:5]))
    return contexts


def format_arguments(arguments: List[EventArgument]) -> List[str]:
    if not arguments:
        return ["- None (no arguments)"]
    lines = []
    for arg in arguments:
        span = f"[{arg.span[0]},{arg.span[1]}]"
        lines.append(f"- {arg.role}: {arg.entity_id} span={span}")
    return lines


def resolve_sent_id(event: EventEntry, doc_ctx: Optional[MavenDocContext]) -> int:
    if doc_ctx is None or not doc_ctx.sentences:
        return 0
    sent_id = doc_ctx.event_sent_ids.get(event.event_id)
    if sent_id is None:
        trigger_text = event.trigger.text.strip().lower()
        if trigger_text and trigger_text in doc_ctx.triggers_by_text:
            for candidate in doc_ctx.triggers_by_text[trigger_text]:
                if isinstance(candidate, int):
                    sent_id = candidate
                    break
    if sent_id is None:
        sent_id = 0
    return max(0, min(sent_id, max(len(doc_ctx.sentences) - 1, 0)))


def build_context_window(
    event: EventEntry,
    doc_ctx: Optional[MavenDocContext],
    window: int,
) -> Tuple[str, int]:
    if doc_ctx is None or not doc_ctx.sentences:
        fallback = event.trigger.text.strip() or "UNKNOWN CONTEXT"
        return fallback, len(fallback.split())
    sent_id = resolve_sent_id(event, doc_ctx)
    sentences = doc_ctx.sentences
    start = max(sent_id - window, 0)
    end = min(sent_id + window + 1, len(sentences))
    context_sentences = [sentences[idx].strip() for idx in range(start, end) if sentences[idx].strip()]
    if not context_sentences and sentences:
        primary = sentences[sent_id].strip()
        if primary:
            context_sentences = [primary]
    if not context_sentences:
        context_sentences = [
            sent.strip() for sent in sentences if isinstance(sent, str) and sent.strip()
        ]
    if not context_sentences:
        context_sentences = [event.trigger.text.strip() or "UNKNOWN CONTEXT"]
    context = " ".join(context_sentences).strip()
    token_count = len(context.split())
    return context, token_count


def validate_sft_sample(sample: SFTSample) -> None:
    required_sections = ["[DOC]", "[CONTEXT]", "[EVENT]", "[TRIGGER]", "[ARGUMENTS]", "[TIME]"]
    for section in required_sections:
        assert section in sample.input, f"input 缺少 {section} 段落"
    assert "[ARGUMENTS]\n" in sample.input, "input 缺少论元列表"
    assert sample.instruction.strip(), "instruction 不能为空"
    assert isinstance(sample.history, list), "history 必须是列表"
    json.loads(sample.output)


def build_sft_samples(
    events: List[EventEntry],
    context_index: Dict[str, MavenDocContext],
    window: int = 1,
) -> Tuple[List[SFTSample], Dict[str, List[SFTSample]]]:
    samples: List[SFTSample] = []
    grouped: Dict[str, List[SFTSample]] = defaultdict(list)
    total_tokens = 0
    no_argument_count = 0
    sorted_events = sorted(events, key=lambda item: (item.doc_id, item.event_id))
    for event in sorted_events:
        if not event.arguments:
            no_argument_count += 1
        doc_ctx = context_index.get(event.doc_id)
        context_text, token_count = build_context_window(event, doc_ctx, window)
        total_tokens += token_count
        time_value = event.time.value if event.time else "UNKNOWN"
        argument_lines = format_arguments(event.arguments)
        prompt_lines = [
            f"[DOC] {event.doc_id}",
            f"[CONTEXT] {context_text}",
            f"[EVENT] {event.event_id}",
            f"[TRIGGER] {event.trigger.text} ({event.trigger.type})",
            "[ARGUMENTS]",
            *argument_lines,
            f"[TIME] {time_value}",
        ]
        prompt = "\n".join(prompt_lines) + "\n"
        response_payload = {
            "trigger": event.trigger.text,
            "type": event.trigger.type,
        }
        if event.arguments:
            response_payload["args"] = {arg.role: arg.entity_id for arg in event.arguments}
        response = json.dumps(response_payload, ensure_ascii=False)
        sample = SFTSample(
            instruction="根据上下文抽取事件的触发词、类型与论元。",
            input=prompt,
            output=response,
            system="你是一名事件抽取助手，请使用 JSON 输出结果。",
        )
        validate_sft_sample(sample)
        samples.append(sample)
        if event.split:
            grouped[event.split].append(sample)
    num_samples = len(samples)
    avg_tokens = (total_tokens / num_samples) if num_samples else 0.0
    no_arg_ratio = (no_argument_count / num_samples) if num_samples else 0.0
    LOGGER.info(
        "构造 SFT 样本 %d 条。无论元比例：%.2f%%，平均上下文 token 数：%.2f",
        num_samples,
        no_arg_ratio * 100,
        avg_tokens,
    )
    if grouped:
        for split_name, split_samples in grouped.items():
            LOGGER.info("  - %s: %d 条", split_name, len(split_samples))
    return samples, grouped


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
        shuffled = list(chunk)
        rng.shuffle(shuffled)
        localized: List[Dict[str, object]] = []
        candidate_map: Dict[str, str] = {}
        for local_idx, traj in enumerate(shuffled, start=1):
            candidate_id = f"C{local_idx:03d}"
            payload = traj.model_dump()
            payload["candidate_id"] = candidate_id
            localized.append(payload)
            candidate_map[candidate_id] = traj.trajectory_id

        window_id = f"maven_window_{window_idx:05d}"
        window = build_window_from_trajectories(
            localized,
            window_id=window_id,
            intent_id="MAVEN-ERE",
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


# =============================
# 主流程
# =============================


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MAVEN raw JSON to processed ECPO datasets")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="原始 JSON 目录")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="输出目录")
    args = parser.parse_args()

    ensure_dir(args.dst)

    skeleton_map = load_mapping(MAPPING_DIR / "event2skeleton.json", default={})
    cameo_map = load_mapping(MAPPING_DIR / "event2cameo.json", default={})

    events = load_events(args.src, skeleton_map, cameo_map)
    if not events:
        LOGGER.error("未生成任何事件，流程终止。")
        return

    filtered_events = [event for event in events if event.source != "MAVEN-CANDIDATE"]
    removed_count = len(events) - len(filtered_events)
    if removed_count:
        LOGGER.info("过滤掉 %d 条仅包含候选触发词的事件。", removed_count)
    events = filtered_events

    doc_ids = {event.doc_id for event in events}
    context_index = build_doc_context_index(args.src, doc_ids)

    event_stats = write_jsonl(args.dst / "event.jsonl", events)

    base_trajs, _ = build_trajectories(events, skeleton_map, context_index)
    extra_trajs, pairs = build_preference_pairs(base_trajs, random.Random(42))
    all_trajs = base_trajs + extra_trajs
    traj_stats = write_jsonl(args.dst / "traj.jsonl", all_trajs)
    pair_stats = write_jsonl(args.dst / "pairs.jsonl", pairs)

    sft_samples, sft_by_split = build_sft_samples(events, context_index)
    sft_stats = write_jsonl(args.dst / "maven_sft.jsonl", sft_samples)
    sft_split_stats: List[DatasetStats] = []
    for split_name, split_samples in sorted(sft_by_split.items()):
        split_path = args.dst / f"maven_sft_{split_name}.jsonl"
        sft_split_stats.append(write_jsonl(split_path, split_samples))

    rl_prompts = build_rl_prompts(all_trajs)
    rl_stats = write_jsonl(args.dst / "rl_prompts.jsonl", rl_prompts)

    write_dataset_info(
        args.dst,
        {
            "maven_sft": {
                "file_name": "maven_sft.jsonl",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history",
                },
            },
            "maven_rl": {
                "file_name": "rl_prompts.jsonl",
                "formatting": "ecpo_rl",
                "columns": {"prompt": "prompt", "response": "response"},
            },
        },
    )

    summary_path = args.dst / "summary.stats.json"
    summary_items = [event_stats, traj_stats, pair_stats, sft_stats]
    summary_items.extend(sft_split_stats)
    summary_items.append(rl_stats)
    build_summary(summary_items, summary_path)
    LOGGER.info("处理完成，结果写入 %s", args.dst)


if __name__ == "__main__":
    main()
