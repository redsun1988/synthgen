from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


def stable_id(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Document(BaseModel):
    """Единица входного корпуса."""

    id: str
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, **meta: Any) -> "Document":
        return cls(id=stable_id(text), text=text, meta=meta)


class Record(BaseModel):
    """Каноническая запись датасета: (текст, схема задачи, типизированный ответ, уверенность, рассуждение).

    Все экспортёры строятся поверх этого формата.
    """

    id: str
    task: str
    lang: Literal["ru", "en"]
    text: str
    task_schema: dict[str, Any] = Field(description="Схема задачи (метки, типы сущностей, колонки и т.д.)")
    answer: dict[str, Any] = Field(description="Итоговый типизированный ответ")
    confidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Мягкие метки / согласие сэмплов: распределение классов, std регрессии, доля голосов за спан",
    )
    reasoning: str | None = None
    source: Literal["corpus", "generated"] = "corpus"
    doc_id: str
    quality: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
