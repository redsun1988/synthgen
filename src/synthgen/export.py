"""Экспорт канонических записей в форматы для обучения."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from synthgen.record import Record
from synthgen.tasks import TASKS
from synthgen.tasks.base import L


def _write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _task_and_schema(r: Record):
    task = TASKS[r.task]()
    return task, task.schema_cls.model_validate(r.task_schema)


def _strip_offsets(obj: Any) -> Any:
    """Ответ для генеративной модели: без служебных смещений и уверенностей."""
    if isinstance(obj, dict):
        return {k: _strip_offsets(v) for k, v in obj.items() if k not in ("start", "end", "confidence")}
    if isinstance(obj, list):
        return [_strip_offsets(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- Jev-подобный формат


def to_structured(records: Iterable[Record]) -> list[dict[str, Any]]:
    """(вход, JSON-схема выхода, типизированный выход, вероятности) — для моделей, отвечающих по схеме."""
    rows = []
    for r in records:
        task, schema = _task_and_schema(r)
        rows.append({
            "id": r.id,
            "task": r.task,
            "instruction": task.instruction(schema, r.lang),
            "input": r.text,
            "output_schema": task.answer_model(schema).model_json_schema(),
            "output": r.answer,
            "probabilities": r.confidence,
        })
    return rows


# --------------------------------------------------------------------------- chat / SFT


def to_chat(records: Iterable[Record], include_reasoning: bool = True, think_tags: tuple[str, str] = ("<think>", "</think>")) -> list[dict[str, Any]]:
    """messages-формат для TRL/Unsloth/axolotl. Рассуждение кладётся в <think>...</think> перед JSON-ответом."""
    rows = []
    for r in records:
        task, schema = _task_and_schema(r)
        messages = task.label_messages(r.text, schema, r.lang)
        answer = json.dumps(_strip_offsets(r.answer), ensure_ascii=False)
        content = f"{think_tags[0]}\n{r.reasoning}\n{think_tags[1]}\n{answer}" if include_reasoning and r.reasoning else answer
        rows.append({"id": r.id, "messages": [*messages, {"role": "assistant", "content": content}]})
    return rows


# --------------------------------------------------------------------------- NER: BIO / CoNLL

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def to_bio(record: Record) -> list[tuple[str, str]]:
    """Токены и BIO-теги. Для вложенных сущностей остаётся внешняя (самая длинная)."""
    entities = sorted(record.answer.get("entities", []), key=lambda e: (e["start"], -(e["end"] - e["start"])))
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(record.text)]
    tags = ["O"] * len(tokens)
    for e in entities:
        idx = [i for i, (_, s, en) in enumerate(tokens) if s >= e["start"] and en <= e["end"]]
        if not idx or any(tags[i] != "O" for i in idx):
            continue
        tags[idx[0]] = f"B-{e['type']}"
        for i in idx[1:]:
            tags[i] = f"I-{e['type']}"
    return [(t, tag) for (t, _, _), tag in zip(tokens, tags)]


def to_conll(records: Iterable[Record], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"# id = {r.id}\n")
            f.writelines(f"{tok}\t{tag}\n" for tok, tag in to_bio(r))
            f.write("\n")
    return path


# --------------------------------------------------------------------------- QA: SQuAD v2


def to_squad(records: Iterable[Record]) -> dict[str, Any]:
    data = []
    for r in records:
        qas = [
            {
                "id": f"{r.id}-{i}",
                "question": qa["question"],
                "is_impossible": not qa["answerable"],
                "answers": [{"text": a["text"], "answer_start": a["start"]} for a in qa["answers"]],
            }
            for i, qa in enumerate(r.answer.get("qas", []))
        ]
        data.append({"title": r.doc_id, "paragraphs": [{"context": r.text, "qas": qas}]})
    return {"version": "v2.0", "data": data}


# --------------------------------------------------------------------------- таблицы


def to_table(records: Iterable[Record], with_confidence: bool = False) -> pd.DataFrame:
    """Строки задачи tabular -> DataFrame с приведёнными типами колонок."""
    records = [r for r in records if r.task == "tabular"]
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        row = {"doc_id": r.doc_id, **r.answer["row"]}
        if with_confidence:
            row.update({f"{k}__conf": v for k, v in r.confidence.get("column_agreement", {}).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in records[0].task_schema["columns"]:
        name, dtype = col["name"], col["dtype"]
        if name not in df:
            continue
        if dtype == "int":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        elif dtype == "float":
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif dtype == "bool":
            df[name] = df[name].astype("boolean")
        elif dtype == "category":
            df[name] = pd.Categorical(df[name], categories=col.get("categories"))
        elif dtype == "date":
            df[name] = pd.to_datetime(df[name], errors="coerce")
    return df


# --------------------------------------------------------------------------- HuggingFace


def to_hf(rows: list[dict[str, Any]]):
    """Список словарей (результат любого экспортёра) -> datasets.Dataset."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install synthgen[hf]") from e
    return HFDataset.from_list(rows)


write_jsonl = _write_jsonl
