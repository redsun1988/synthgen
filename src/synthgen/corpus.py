from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np

from synthgen.record import Document, stable_id

_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
_LATIN = re.compile(r"[a-zA-Z]")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"«(\[A-ZА-ЯЁ0-9])")


def detect_lang(text: str) -> Literal["ru", "en"]:
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    return "ru" if cyr >= lat else "en"


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Границы предложений как пары (start, end) символьных смещений."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


# --------------------------------------------------------------------------- адаптеры


def read_jsonl(path: str | Path, text_field: str = "text", id_field: str | None = "id") -> Iterator[Document]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.pop(text_field)
            doc_id = str(row.pop(id_field)) if id_field and id_field in row else stable_id(text)
            yield Document(id=doc_id, text=text, meta=row)


def read_txt_dir(path: str | Path, pattern: str = "*.txt") -> Iterator[Document]:
    for file in sorted(Path(path).rglob(pattern)):
        yield Document(id=file.stem, text=file.read_text(encoding="utf-8"), meta={"path": str(file)})


def from_texts(texts: Iterable[str]) -> list[Document]:
    return [Document.from_text(t) for t in texts]


# --------------------------------------------------------------------------- нарезка


def chunk_documents(
    docs: Iterable[Document], max_chars: int = 1500, min_chars: int = 200
) -> Iterator[Document]:
    """Режет документы на чанки по границам предложений. Чанк помнит исходный документ."""
    for doc in docs:
        text = doc.text.strip()
        if len(text) <= max_chars:
            if len(text) >= min_chars:
                yield doc
            continue
        buf_start: int | None = None
        buf_end = 0
        part = 0
        for s, e in split_sentences(text):
            if buf_start is not None and e - buf_start > max_chars:
                chunk = text[buf_start:buf_end]
                if len(chunk) >= min_chars:
                    yield Document(id=f"{doc.id}#{part}", text=chunk, meta={**doc.meta, "parent_id": doc.id})
                    part += 1
                buf_start = None
            if buf_start is None:
                buf_start = s
            buf_end = e
        if buf_start is not None and len(text[buf_start:buf_end]) >= min_chars:
            yield Document(id=f"{doc.id}#{part}", text=text[buf_start:buf_end], meta={**doc.meta, "parent_id": doc.id})


# --------------------------------------------------------------------------- разнообразие


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """Простой k-means (k-means++ инициализация) без зависимостей от sklearn."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(x[rng.choice(len(x), p=d / d.sum()) if d.sum() > 0 else rng.integers(len(x))])
    c = np.asarray(centers)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        labels = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new_c = np.asarray([x[labels == j].mean(0) if (labels == j).any() else c[j] for j in range(k)])
        if np.allclose(new_c, c):
            break
        c = new_c
    return labels


def select_diverse(docs: list[Document], embeddings: np.ndarray, k: int, seed: int = 0) -> list[Document]:
    """Выбирает k документов, покрывающих разные кластеры корпуса (ближайший к центру каждого кластера)."""
    if k >= len(docs):
        return list(docs)
    labels = kmeans(embeddings, k, seed=seed)
    chosen: list[Document] = []
    for j in range(k):
        idx = np.flatnonzero(labels == j)
        if len(idx) == 0:
            continue
        center = embeddings[idx].mean(0)
        chosen.append(docs[idx[np.argmax(embeddings[idx] @ center)]])
    return chosen
