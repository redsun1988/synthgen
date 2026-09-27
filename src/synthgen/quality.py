"""Контроль качества: LLM-судья, дедупликация (MinHash + эмбеддинги), баланс меток."""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from synthgen.llm import LLMClient
from synthgen.record import Record
from synthgen.tasks.base import L, Lang

# --------------------------------------------------------------------------- LLM-судья


class JudgeVerdict(BaseModel):
    issues: list[str] = Field(description="Конкретные ошибки разметки; пусто, если их нет")
    score: int = Field(ge=1, le=5, description="1 — разметка неверна, 3 — есть спорные места, 5 — безупречна")


async def judge(llm: LLMClient, instruction: str, text: str, answer: dict[str, Any], lang: Lang) -> JudgeVerdict:
    answer_json = json.dumps(answer, ensure_ascii=False)
    prompt = L(
        lang,
        f"Проверь разметку, сделанную другим аннотатором. Сначала перечисли конкретные ошибки (пропуски, "
        f"лишнее, неверные метки, выдумки), затем поставь оценку 1-5.\n\nЗАДАНИЕ АННОТАТОРУ:\n{instruction}\n\n"
        f"ТЕКСТ:\n\"\"\"\n{text}\n\"\"\"\n\nРАЗМЕТКА:\n{answer_json}",
        f"Review the annotation made by another annotator. First list concrete errors (misses, extras, wrong "
        f"labels, hallucinations), then give a 1-5 score.\n\nANNOTATOR TASK:\n{instruction}\n\n"
        f"TEXT:\n\"\"\"\n{text}\n\"\"\"\n\nANNOTATION:\n{answer_json}",
    )
    return await llm.structured([{"role": "user", "content": prompt}], JudgeVerdict, temperature=0.0)


# --------------------------------------------------------------------------- MinHash


_WORD = re.compile(r"\w+", re.UNICODE)
_PRIME = (1 << 61) - 1


def _shingles(text: str, n: int = 3) -> set[int]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {zlib.crc32(" ".join(words).encode())}
    return {zlib.crc32(" ".join(words[i : i + n]).encode()) for i in range(len(words) - n + 1)}


def minhash_signatures(texts: Sequence[str], num_perm: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, num_perm, dtype=np.uint64)
    b = rng.integers(0, _PRIME, num_perm, dtype=np.uint64)
    sigs = np.empty((len(texts), num_perm), dtype=np.uint64)
    for i, t in enumerate(texts):
        sh = np.fromiter(_shingles(t), dtype=np.uint64)
        # (a*x + b) mod p по всем перестановкам сразу; переполнение uint64 допустимо для хеширования
        sigs[i] = ((np.outer(sh, a) + b) % _PRIME).min(axis=0)
    return sigs


def near_duplicates_minhash(texts: Sequence[str], threshold: float = 0.7, num_perm: int = 64, bands: int = 16) -> set[int]:
    """Индексы текстов, являющихся почти-дубликатами более ранних (LSH по полосам сигнатур)."""
    sigs = minhash_signatures(texts, num_perm)
    rows = num_perm // bands
    buckets: dict[tuple[int, bytes], list[int]] = {}
    dup: set[int] = set()
    for i, sig in enumerate(sigs):
        candidates: set[int] = set()
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows].tobytes())
            candidates.update(buckets.setdefault(key, []))
            buckets[key].append(i)
        if any(j not in dup and (sigs[j] == sig).mean() >= threshold for j in candidates):
            dup.add(i)
    return dup


def near_duplicates_embeddings(embeddings: np.ndarray, threshold: float = 0.95) -> set[int]:
    sim = embeddings @ embeddings.T
    dup: set[int] = set()
    for i in range(len(embeddings)):
        if any(sim[i, j] >= threshold and j not in dup for j in range(i)):
            dup.add(i)
    return dup


async def deduplicate(
    records: list[Record],
    minhash_threshold: float = 0.7,
    llm: LLMClient | None = None,
    embedding_threshold: float = 0.95,
) -> tuple[list[Record], dict[str, int]]:
    """Убирает почти-дубликаты текстов. Эмбеддинги используются, если передан llm с embed_model."""
    texts = [r.text for r in records]
    dup = near_duplicates_minhash(texts, minhash_threshold)
    stats = {"minhash": len(dup), "embedding": 0}
    if llm is not None and llm.config.embed_model:
        emb = await llm.embed(texts)
        emb_dup = near_duplicates_embeddings(emb, embedding_threshold) - dup
        stats["embedding"] = len(emb_dup)
        dup |= emb_dup
    return [r for i, r in enumerate(records) if i not in dup], stats


# --------------------------------------------------------------------------- баланс


def label_distribution(records: Sequence[Record]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for r in records:
        if "label" in r.answer:
            counts[r.answer["label"]] += 1
        for lab in r.answer.get("labels", []):
            counts[lab] += 1
    return counts


def imbalance_ratio(counts: Counter[str], labels: Sequence[str]) -> float:
    values = [counts.get(lab, 0) for lab in labels]
    return max(values) / max(min(values), 1)
