from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from synthgen.record import Record


class Dataset:
    """Список канонических записей с сохранением, статистикой и разбиением без утечек."""

    def __init__(self, records: Iterable[Record] = ()):
        self.records: list[Record] = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __getitem__(self, i: int) -> Record:
        return self.records[i]

    def __repr__(self) -> str:
        return f"Dataset({len(self)} records, tasks={dict(Counter(r.task for r in self.records))})"

    # ------------------------------------------------------------------ io

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(r.model_dump_json() + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        with open(path, encoding="utf-8") as f:
            return cls(Record.model_validate_json(line) for line in f if line.strip())

    def to_pandas(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({
                "id": r.id, "task": r.task, "lang": r.lang, "source": r.source, "doc_id": r.doc_id,
                "schema": r.task_schema.get("name"), "text": r.text,
                "answer": json.dumps(r.answer, ensure_ascii=False),
                "confidence": json.dumps(r.confidence, ensure_ascii=False),
                "judge_score": r.quality.get("judge_score"), "has_reasoning": r.reasoning is not None,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ операции

    def filter(self, predicate) -> "Dataset":
        return Dataset(r for r in self.records if predicate(r))

    def __add__(self, other: "Dataset") -> "Dataset":
        return Dataset([*self.records, *other.records])

    def split(self, val: float = 0.1, test: float = 0.1, seed: int = 0) -> dict[str, "Dataset"]:
        """Разбиение по исходному документу: чанки и производные одного документа не попадают в разные части."""
        groups = sorted({self._group(r) for r in self.records})
        random.Random(seed).shuffle(groups)
        n_test, n_val = round(len(groups) * test), round(len(groups) * val)
        part = {g: "test" for g in groups[:n_test]} | {g: "val" for g in groups[n_test : n_test + n_val]}
        out: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
        for r in self.records:
            out[part.get(self._group(r), "train")].append(r)
        return {k: Dataset(v) for k, v in out.items()}

    @staticmethod
    def _group(r: Record) -> str:
        return r.meta.get("parent_id") or r.doc_id.split("#")[0]

    def stats(self) -> dict:
        return {
            "records": len(self),
            "by_task": dict(Counter(r.task for r in self.records)),
            "by_lang": dict(Counter(r.lang for r in self.records)),
            "by_source": dict(Counter(r.source for r in self.records)),
            "schemas": len({json.dumps(r.task_schema, sort_keys=True) for r in self.records}),
            "with_reasoning": sum(r.reasoning is not None for r in self.records),
        }
