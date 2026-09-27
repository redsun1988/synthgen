from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, create_model

from synthgen.llm import LLMClient, Messages
from synthgen.schemas import TaskSchema

S = TypeVar("S", bound=TaskSchema)
Lang = Literal["ru", "en"]


def L(lang: Lang, ru: str, en: str) -> str:
    return ru if lang == "ru" else en


def schema_json(schema: BaseModel) -> str:
    return json.dumps(schema.model_dump(exclude_none=True), ensure_ascii=False, indent=1)


def literal_of(values: list[str]) -> Any:
    return Literal[tuple(values)]  # type: ignore[valid-type]


class Task(ABC, Generic[S]):
    """Тип задачи: как размечать текст по схеме, как проверять ответ и сводить несколько сэмплов.

    Жизненный цикл одного примера:
    label_messages -> LLM(answer_model) -> validate (с самоисправлением) -> postprocess -> aggregate(n сэмплов)
    """

    name: ClassVar[str]
    schema_cls: ClassVar[type[TaskSchema]]
    # Какую задачу придумывать в режиме auto-schema.
    proposal_hint: ClassVar[tuple[str, str]] = ("", "")

    # ------------------------------------------------------------------ разметка

    @abstractmethod
    def answer_model(self, schema: S) -> type[BaseModel]:
        """Pydantic-модель ответа LLM (с Literal-ами меток, чтобы сервер ограничил генерацию)."""

    @abstractmethod
    def instruction(self, schema: S, lang: Lang) -> str:
        """Описание задачи для модели-разметчика. Используется и в chat/SFT-экспорте как user-промпт."""

    def label_messages(self, text: str, schema: S, lang: Lang) -> Messages:
        system = L(
            lang,
            "Ты аккуратный аннотатор данных. Отвечай строго JSON по заданной схеме.",
            "You are a careful data annotator. Answer strictly with JSON matching the schema.",
        )
        user = f"{self.instruction(schema, lang)}\n\n{L(lang, 'Текст', 'Text')}:\n\"\"\"\n{text}\n\"\"\""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def validate(self, text: str, schema: S, raw: BaseModel) -> list[str]:
        return []

    def repair(self, text: str, schema: S, raw: BaseModel) -> BaseModel | None:
        """Отбрасывает невалидные части ответа, если самоисправление не помогло. None — чинить нельзя."""
        return None

    @abstractmethod
    def postprocess(self, text: str, schema: S, raw: BaseModel) -> dict[str, Any]:
        """Приводит ответ LLM к каноническому виду (например, добавляет символьные смещения)."""

    @abstractmethod
    def aggregate(self, text: str, schema: S, answers: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Сводит n сэмплов в итоговый ответ и мягкие метки/уверенность."""

    async def label(
        self, llm: LLMClient, text: str, schema: S, lang: Lang, n_samples: int = 1, temperature: float = 0.7
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model = self.answer_model(schema)
        messages = self.label_messages(text, schema, lang)
        temp = temperature if n_samples > 1 else min(temperature, 0.2)
        repair = (lambda r: self.repair(text, schema, r)) if type(self).repair is not Task.repair else None
        results = await asyncio.gather(
            *[
                llm.structured(messages, model, temperature=temp, repair=repair,
                               validator=lambda r: self.validate(text, schema, r))
                for _ in range(n_samples)
            ],
            return_exceptions=True,
        )
        raws = [r for r in results if not isinstance(r, BaseException)]
        if not raws:
            raise next(r for r in results if isinstance(r, BaseException))
        # Мягкие метки считаются по успешным сэмплам; сколько их было — видно в confidence.
        answers = [self.postprocess(text, schema, r) for r in raws]
        answer, confidence = self.aggregate(text, schema, answers)
        if len(raws) < n_samples:
            confidence["valid_samples"] = len(raws)
        return answer, confidence

    # ------------------------------------------------------------------ режим generate

    def generation_targets(self, schema: S) -> list[Any]:
        """Цели, по которым балансируется генерация новых текстов (метки классов, значения регрессии)."""
        return [None]

    def generation_hint(self, schema: S, target: Any, lang: Lang) -> str:
        return L(lang, f"подходит для задачи: {schema.instruction}", f"is suitable for the task: {schema.instruction}")

    def target_matches(self, schema: S, target: Any, answer: dict[str, Any]) -> bool:
        return True

    # ------------------------------------------------------------------ auto-schema

    def proposal_messages(self, texts: list[str], k: int, lang: Lang) -> Messages:
        ru_hint, en_hint = self.proposal_hint
        samples = "\n\n".join(f"[{i + 1}] {t[:1500]}" for i, t in enumerate(texts))
        user = L(
            lang,
            f"Ниже примеры текстов из корпуса. Придумай {k} разных, полезных и однозначно размечаемых задач типа "
            f"«{self.name}» для этих текстов. {ru_hint}\nЗадачи должны быть разнообразными, решаемыми по самому тексту "
            f"и применимыми к любым похожим текстам того же жанра, а не только к приведённым. Каждой задаче дай "
            f"собственное осмысленное имя. Описания пиши на русском.\n\n{samples}",
            f"Below are sample texts from a corpus. Invent {k} different, useful, unambiguously annotatable tasks of type "
            f"'{self.name}' for these texts. {en_hint}\nTasks must be diverse, solvable from the text itself and "
            f"applicable to any similar text of the same genre, not only to the samples. Give each task its own "
            f"meaningful name. Write descriptions in English.\n\n{samples}",
        )
        return [{"role": "user", "content": user}]

    async def propose_schemas(self, llm: LLMClient, texts: list[str], k: int, lang: Lang) -> list[S]:
        wrapper = create_model(f"{self.schema_cls.__name__}List", schemas=(list[self.schema_cls], ...))  # type: ignore[valid-type]

        def check(r: Any) -> list[str]:
            errors = [] if len(r.schemas) >= k else [f"Нужно ровно {k} схем, получено {len(r.schemas)}"]
            for s in r.schemas:
                errors += self.check_schema(s)
            return errors

        result = await llm.structured(self.proposal_messages(texts, k, lang), wrapper, temperature=0.9, validator=check)
        return list(result.schemas)[:k]  # type: ignore[attr-defined]

    def check_schema(self, schema: S) -> list[str]:
        return []
