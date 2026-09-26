from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator


class EventArgument(BaseModel):
    role: str
    entity_id: str
    span: List[int]

    @field_validator("span")
    @classmethod
    def _validate_span(cls, value: List[int]) -> List[int]:
        assert len(value) == 2 and value[0] <= value[1], "span must be [start, end]"
        return value


class EventTrigger(BaseModel):
    span: List[int]
    text: str
    type: str

    @field_validator("span")
    @classmethod
    def _validate_trigger_span(cls, value: List[int]) -> List[int]:
        assert len(value) == 2 and value[0] <= value[1], "trigger span must be [start, end]"
        return value


class EventRelation(BaseModel):
    type: str
    head: str
    tail: str


class EventRelations(BaseModel):
    temporal: List[EventRelation]
    causal: List[EventRelation]
    subevent: List[EventRelation]


class EventTime(BaseModel):
    value: str
    span: List[int]

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("span")
    @classmethod
    def _validate_span(cls, value: List[int]) -> List[int]:
        assert len(value) == 2 and value[0] <= value[1], "time span must be [start, end]"
        return value


class EventConfidence(BaseModel):
    trigger_prob: float = Field(..., ge=0.0, le=1.0)
    arg_role_avg: float = Field(..., ge=0.0, le=1.0)


class EventEntry(BaseModel):
    doc_id: str
    event_id: str
    trigger: EventTrigger
    arguments: List[EventArgument]
    time: EventTime
    relations: EventRelations
    confidence: EventConfidence
    source: str
    mapping: Dict[str, str]
    split: Optional[str] = None


class TrajectoryStep(BaseModel):
    event_id: str
    time: str
    type: str
    roles: Dict[str, str]
    delta_days_from_prev: int
    text_refs: List[Dict[str, object]]
    skeleton_hits: List[str]

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value


class TrajectoryMeta(BaseModel):
    graph_nodes: List[str]
    graph_edges: List[List[str]]


class TrajectoryEntry(BaseModel):
    person_id: str
    trajectory_id: str
    label: str
    steps: List[TrajectoryStep]
    meta: TrajectoryMeta

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        assert value in {"expert", "candidate", "negative"}
        return value

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: List[TrajectoryStep]) -> List[TrajectoryStep]:
        assert value, "trajectory must have at least one step"
        return value


class PreferencePair(BaseModel):
    better: str
    worse: str
    reason: str


class SFTSample(BaseModel):
    instruction: str
    input: str
    output: str
    system: str
    history: List[List[str]] = Field(default_factory=list)


class RLPrompt(BaseModel):
    prompt: str
    response: str
    trajectory_id: str
    person_id: str
    window_id: Optional[str] = None
    intent_id: Optional[str] = None
    candidate_ids: List[str] = Field(default_factory=list)
    candidate_map: Dict[str, str] = Field(default_factory=dict)
    skeleton_steps: List[str] = Field(default_factory=list)


@dataclass
class DatasetStats:
    name: str
    num_records: int
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "num_records": self.num_records, **self.extra}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[BaseModel | Dict]) -> DatasetStats:
    records = list(records)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            if isinstance(item, BaseModel):
                payload = item.model_dump(exclude_none=True)
            else:
                payload = item
            if path.name.endswith("rl_prompts.jsonl"):
                meta = {
                    "trajectory_id": payload.get("trajectory_id"),
                    "person_id": payload.get("person_id"),
                    "window_id": payload.get("window_id"),
                    "intent_id": payload.get("intent_id"),
                    "candidate_ids": payload.get("candidate_ids"),
                    "candidate_map": payload.get("candidate_map"),
                    "skeleton_steps": payload.get("skeleton_steps"),
                }
                payload["_meta"] = {k: v for k, v in meta.items() if v}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    extra: Dict[str, object] = {}
    if path.name.endswith("event.jsonl"):
        doc_ids = {r.doc_id for r in records if isinstance(r, EventEntry)}
        extra["num_docs"] = len(doc_ids)
    if path.name.endswith("traj.jsonl"):
        labels: Dict[str, int] = {}
        lengths: List[int] = []
        for r in records:
            if isinstance(r, TrajectoryEntry):
                labels[r.label] = labels.get(r.label, 0) + 1
                lengths.append(len(r.steps))
        extra["label_distribution"] = labels
        extra["avg_len"] = sum(lengths) / len(lengths) if lengths else 0
    if path.name.endswith("pairs.jsonl"):
        unique_ids = set()
        for r in records:
            if isinstance(r, PreferencePair):
                unique_ids.add(r.better)
                unique_ids.add(r.worse)
        extra["num_unique_ids"] = len(unique_ids)
    stats = DatasetStats(name=path.name, num_records=len(records), extra=extra)
    stats_path = path.with_suffix(path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def build_summary(stats: List[DatasetStats], output_path: Path) -> None:
    summary = {item.name: item.to_dict() for item in stats}
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataset_info(dst: Path, entries: Dict[str, Dict[str, object]]) -> None:
    dataset_info_path = dst / "dataset_info.json"
    dataset_info: Dict[str, Dict[str, object]] = {}
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_info.update(entries)
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
