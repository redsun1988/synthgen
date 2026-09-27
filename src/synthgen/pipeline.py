from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from tqdm.auto import tqdm

from synthgen.corpus import detect_lang
from synthgen.dataset import Dataset
from synthgen.llm import LLMClient, LLMError
from synthgen.quality import judge
from synthgen.reasoning import explain
from synthgen.record import Document, Record, stable_id
from synthgen.schemas import TaskSchema
from synthgen.tasks import Task, get_task
from synthgen.tasks.base import L, Lang

logger = logging.getLogger(__name__)

SchemaSpec = TaskSchema | Sequence[TaskSchema] | Literal["auto"]


@dataclass
class GenerationConfig:
    mode: Literal["label", "generate"] = "label"
    """label — размечать тексты корпуса; generate — писать новые тексты по мотивам корпуса и размечать их."""
    n_samples: int = 1
    """Сколько независимых разметок делать на пример; >1 даёт мягкие метки/уверенность."""
    temperature: float = 0.7
    per_document: int = 1
    """generate: сколько новых текстов писать на один документ-образец."""
    reasoning: bool = False
    judge: bool = False
    min_judge_score: int = 3
    rebalance_rounds: int = 2
    """generate: сколько раундов догенерации для недобравших классов/значений."""
    seed: int = 0


@dataclass
class _Job:
    doc: Document
    schema: TaskSchema | None  # None -> придумать схему по документу (auto)
    variant: int
    target: Any = None

    def key(self, task: str, mode: str) -> str:
        schema_key = self.schema.model_dump_json() if self.schema else "auto"
        return stable_id(task, mode, schema_key, self.doc.id, self.variant, self.target)


class Generator:
    """Пайплайн генерации: документы -> (схема) -> (новый текст) -> разметка -> проверки -> рассуждение -> Record.

    Записи сразу дописываются в `output_path` (JSONL), поэтому прерванный запуск продолжается с места остановки.
    """

    def __init__(
        self,
        llm: LLMClient,
        task: Task | str,
        schema: SchemaSpec,
        config: GenerationConfig | None = None,
        output_path: str | Path | None = None,
        **overrides: Any,
    ):
        self.llm = llm
        self.task = get_task(task) if isinstance(task, str) else task
        self.schemas: list[TaskSchema] | None = None if schema == "auto" else (
            [schema] if isinstance(schema, TaskSchema) else list(schema)  # type: ignore[arg-type]
        )
        self.config = replace(config or GenerationConfig(), **overrides)
        self.output_path = Path(output_path) if output_path else None
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------------ public

    async def run(self, docs: Iterable[Document], limit: int | None = None, progress: bool = True) -> Dataset:
        docs = list(itertools.islice(docs, limit))
        done = self._load_existing()
        records: list[Record] = list(done.values())
        jobs = [j for j in self._plan(docs) if j.key(self.task.name, self.config.mode) not in done]
        self.stats["resumed"] = len(records)

        records += await self._execute(jobs, progress, desc=self.task.name)

        if self.config.mode == "generate":
            for round_ in range(1, self.config.rebalance_rounds + 1):
                extra = self._rebalance_jobs(records, docs, round_)
                if not extra:
                    break
                logger.info("Ребалансировка, раунд %d: %d задач", round_, len(extra))
                records += await self._execute(extra, progress, desc=f"rebalance {round_}")
        return Dataset(records)

    def run_sync(self, docs: Iterable[Document], limit: int | None = None, progress: bool = True) -> Dataset:
        return asyncio.run(self.run(docs, limit, progress))

    # ------------------------------------------------------------------ планирование

    def _plan(self, docs: list[Document]) -> list[_Job]:
        jobs: list[_Job] = []
        per_doc = self.config.per_document if self.config.mode == "generate" else 1
        target_cursor: Counter[int] = Counter()
        for i, doc in enumerate(docs):
            schema = self.schemas[i % len(self.schemas)] if self.schemas else None
            for v in range(per_doc):
                target = None
                if self.config.mode == "generate" and schema is not None:
                    targets = self.task.generation_targets(schema)
                    sid = id(schema)
                    target = targets[target_cursor[sid] % len(targets)]
                    target_cursor[sid] += 1
                jobs.append(_Job(doc, schema, v, target))
        return jobs

    def _rebalance_jobs(self, records: list[Record], docs: list[Document], round_: int) -> list[_Job]:
        rng = random.Random(self.config.seed + round_)
        jobs: list[_Job] = []
        for schema in self.schemas or []:
            targets = self.task.generation_targets(schema)
            if targets == [None]:
                continue
            dump = schema.model_dump()
            counts = Counter(
                json.dumps(r.quality.get("target")) for r in records if r.task_schema == dump and r.source == "generated"
            )
            top = max((counts.get(json.dumps(t), 0) for t in targets), default=0)
            for t in targets:
                for k in range(top - counts.get(json.dumps(t), 0)):
                    jobs.append(_Job(rng.choice(docs), schema, 1000 * round_ + k, t))
        return jobs

    # ------------------------------------------------------------------ исполнение

    async def _execute(self, jobs: list[_Job], progress: bool, desc: str) -> list[Record]:
        if not jobs:
            return []
        queue: asyncio.Queue[_Job] = asyncio.Queue()
        for j in jobs:
            queue.put_nowait(j)
        results: list[Record] = []
        bar = tqdm(total=len(jobs), desc=desc, disable=not progress)

        async def worker() -> None:
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await self._process(job)
                except LLMError as e:
                    self.stats["failed"] += 1
                    logger.warning("Документ %s пропущен: %s", job.doc.id, str(e)[:300])
                    record = None
                except Exception:
                    self.stats["failed"] += 1
                    logger.exception("Ошибка на документе %s", job.doc.id)
                    record = None
                if record is not None:
                    results.append(record)
                    self._append(record)
                bar.update(1)
                bar.set_postfix(ok=self.stats["ok"], rejected=self.stats["rejected_target"] + self.stats["rejected_judge"])

        # Воркеров больше лимита запросов: у одного примера бывает несколько последовательных вызовов.
        n_workers = min(len(jobs), self.llm.config.max_concurrency * 2)
        await asyncio.gather(*[worker() for _ in range(n_workers)])
        bar.close()
        return results

    async def _process(self, job: _Job) -> Record | None:
        cfg = self.config
        lang: Lang = detect_lang(job.doc.text)
        schema = job.schema or (await self.task.propose_schemas(self.llm, [job.doc.text], 1, lang))[0]

        text, source = job.doc.text, "corpus"
        if cfg.mode == "generate":
            text, source = await self._write_text(job.doc.text, schema, job.target, lang), "generated"

        answer, confidence = await self.task.label(self.llm, text, schema, lang, cfg.n_samples, cfg.temperature)
        quality: dict[str, Any] = {}
        if cfg.mode == "generate" and job.target is not None:
            quality["target"] = job.target
            if not self.task.target_matches(schema, job.target, answer):
                self.stats["rejected_target"] += 1
                return None

        instruction = self.task.instruction(schema, lang)
        if cfg.judge:
            verdict = await judge(self.llm, instruction, text, answer, lang)
            quality.update(judge_score=verdict.score, judge_issues=verdict.issues)
            if verdict.score < cfg.min_judge_score:
                self.stats["rejected_judge"] += 1
                return None

        reasoning = await explain(self.llm, instruction, text, answer, lang) if cfg.reasoning else None
        self.stats["ok"] += 1
        return Record(
            id=stable_id(job.key(self.task.name, cfg.mode), text),
            task=self.task.name,
            lang=lang,
            text=text,
            task_schema=schema.model_dump(),
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            source=source,
            doc_id=job.doc.id,
            quality=quality,
            meta={
                "key": job.key(self.task.name, cfg.mode),
                "parent_id": job.doc.meta.get("parent_id", job.doc.id.split("#")[0]),
                "model": self.llm.config.model,
            },
        )

    async def _write_text(self, seed_text: str, schema: TaskSchema, target: Any, lang: Lang) -> str:
        hint = self.task.generation_hint(schema, target, lang)
        prompt = L(
            lang,
            f"Ниже образец текста из корпуса. Используй его только как ориентир по домену, жанру, стилю и длине.\n"
            f"Напиши НОВЫЙ реалистичный текст на русском (другие детали, имена, числа, ситуация), который {hint}. "
            f"Длина примерно {len(seed_text)} символов. Верни только сам текст, без заголовков и пояснений.\n\n"
            f"ОБРАЗЕЦ:\n\"\"\"\n{seed_text}\n\"\"\"",
            f"Below is a sample text from a corpus. Use it only as a reference for domain, genre, style and length.\n"
            f"Write a NEW realistic English text (different details, names, numbers, situation) that {hint}. "
            f"About {len(seed_text)} characters long. Return only the text itself, no titles or comments.\n\n"
            f"SAMPLE:\n\"\"\"\n{seed_text}\n\"\"\"",
        )
        return (await self.llm.text([{"role": "user", "content": prompt}], temperature=0.9)).strip().strip('"')

    # ------------------------------------------------------------------ чекпоинты

    def _load_existing(self) -> dict[str, Record]:
        if not self.output_path or not self.output_path.exists():
            return {}
        done = {}
        for r in Dataset.from_jsonl(self.output_path):
            done[r.meta.get("key", r.id)] = r
        return done

    def _append(self, record: Record) -> None:
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
