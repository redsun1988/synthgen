from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, create_model

from synthgen.llm import LLMClient
from synthgen.schemas import QASchema, SummarizationSchema, TabularSchema
from synthgen.spans import char_jaccard, find_first, overlap_coefficient
from synthgen.tasks.base import L, Lang, Task, literal_of

AGREEMENT_THRESHOLD = 0.5


def _spans_with_offsets(text: str, fragments: list[str]) -> list[dict[str, Any]]:
    result = []
    for frag in fragments:
        hit = find_first(text, frag)
        if hit:
            result.append({"start": hit[0], "end": hit[1], "text": text[hit[0] : hit[1]]})
    return result


def _missing(text: str, fragments: list[str]) -> list[str]:
    return [f"Фрагмент «{f[:80]}» не найден в тексте дословно — копируй текст без изменений" for f in fragments if not find_first(text, f)]


# =========================================================================== Extractive QA


class QAItem(BaseModel):
    question: str
    answerable: bool = Field(description="false, если ответа в тексте нет")
    answers: list[str] = Field(description="Дословные фрагменты текста, составляющие ответ; пусто, если answerable=false")


class QAAnswer(BaseModel):
    items: list[QAItem]


class ReaderAnswer(BaseModel):
    answerable: bool
    answers: list[str]


class ExtractiveQA(Task[QASchema]):
    """Генерация пар вопрос-ответ. При n_samples > 1 каждый вопрос независимо проверяется «читателем»,
    доля согласных ответов становится уверенностью, несогласованные вопросы отбрасываются."""

    name = "extractive_qa"
    schema_cls = QASchema
    proposal_hint = (
        "Каждая задача задаёт тематический фокус вопросов (question_focus), важный для этих текстов.",
        "Each task sets a topical question_focus that matters for these texts.",
    )

    def answer_model(self, schema):
        return QAAnswer

    def _n_unanswerable(self, schema: QASchema) -> int:
        return round(schema.n_questions * schema.unanswerable_ratio)

    def instruction(self, schema, lang):
        n_un = self._n_unanswerable(schema)
        focus = f"\n{L(lang, 'Фокус вопросов', 'Question focus')}: {schema.question_focus}" if schema.question_focus else ""
        multi = (
            L(lang, "Ответ может состоять из нескольких фрагментов (например, перечисление).",
              "An answer may consist of several fragments (e.g. an enumeration).")
            if schema.allow_multi_span
            else L(lang, "Ответ — ровно один непрерывный фрагмент.", "An answer is exactly one contiguous fragment.")
        )
        return L(
            lang,
            f"{schema.instruction}{focus}\nСоставь {schema.n_questions} вопросов по тексту. Из них {n_un} должны быть "
            f"правдоподобными, но без ответа в тексте (answerable=false, answers=[]). Остальные — с ответом, "
            f"который дословно копируется из текста, как можно короче. {multi}\nНе задавай вопросы, ответ на которые "
            f"требует внешних знаний. Вопросы пиши на языке текста.",
            f"{schema.instruction}{focus}\nWrite {schema.n_questions} questions about the text. {n_un} of them must be "
            f"plausible but NOT answerable from the text (answerable=false, answers=[]). The rest must have an answer "
            f"copied verbatim from the text, as short as possible. {multi}\nDo not ask questions requiring outside "
            f"knowledge. Write questions in the language of the text.",
        )

    def validate(self, text, schema, raw):
        errors: list[str] = []
        items = raw.items  # type: ignore[attr-defined]
        if len(items) != schema.n_questions:
            errors.append(f"Нужно ровно {schema.n_questions} вопросов, получено {len(items)}")
        n_un = sum(not it.answerable for it in items)
        if n_un != self._n_unanswerable(schema):
            errors.append(f"Вопросов без ответа должно быть {self._n_unanswerable(schema)}, получено {n_un}")
        for it in items:
            if it.answerable and not it.answers:
                errors.append(f"У вопроса «{it.question}» answerable=true, но answers пуст")
            if not it.answerable and it.answers:
                errors.append(f"У вопроса «{it.question}» answerable=false, но answers не пуст")
            if not schema.allow_multi_span and len(it.answers) > 1:
                errors.append(f"У вопроса «{it.question}» должен быть один фрагмент ответа")
            errors += _missing(text, it.answers)
        return errors

    def repair(self, text, schema, raw):
        # Количество вопросов не чиним: оставляем только корректные пары вопрос-ответ.
        items = [
            it for it in raw.items  # type: ignore[attr-defined]
            if it.answerable == bool(it.answers) and not _missing(text, it.answers)
            and (schema.allow_multi_span or len(it.answers) <= 1)
        ]
        return raw.model_copy(update={"items": items}) if items else None

    def postprocess(self, text, schema, raw):
        return {
            "qas": [
                {"question": it.question, "answerable": it.answerable, "answers": _spans_with_offsets(text, it.answers)}
                for it in raw.items  # type: ignore[attr-defined]
            ]
        }

    def aggregate(self, text, schema, answers):
        return answers[0], {}

    async def label(self, llm: LLMClient, text, schema, lang, n_samples=1, temperature=0.7):
        raw = await llm.structured(
            self.label_messages(text, schema, lang), QAAnswer, temperature=temperature,
            validator=lambda r: self.validate(text, schema, r), repair=lambda r: self.repair(text, schema, r),
        )
        answer = self.postprocess(text, schema, raw)
        if n_samples <= 1:
            return answer, {}
        agreements = await asyncio.gather(*[self._verify(llm, text, qa, lang, n_samples) for qa in answer["qas"]])
        kept = []
        for qa, agr in zip(answer["qas"], agreements):
            qa["confidence"] = round(agr, 3)
            if agr >= AGREEMENT_THRESHOLD:
                kept.append(qa)
        return {"qas": kept}, {"agreement": [round(a, 3) for a in agreements], "dropped": len(answer["qas"]) - len(kept)}

    async def _verify(self, llm: LLMClient, text: str, qa: dict[str, Any], lang: Lang, n: int) -> float:
        prompt = L(
            lang,
            f"Ответь на вопрос строго по тексту. Ответ — минимальные дословные фрагменты текста, без лишних слов. Если ответа в тексте нет, "
            f"answerable=false и answers=[].\n\nТекст:\n\"\"\"\n{text}\n\"\"\"\n\nВопрос: {qa['question']}",
            f"Answer the question strictly from the text using minimal verbatim fragments, no extra words. If the text has no answer, "
            f"answerable=false and answers=[].\n\nText:\n\"\"\"\n{text}\n\"\"\"\n\nQuestion: {qa['question']}",
        )
        gold = [(a["start"], a["end"]) for a in qa["answers"]]
        reads = await asyncio.gather(
            *[
                llm.structured([{"role": "user", "content": prompt}], ReaderAnswer, temperature=0.7,
                               validator=lambda r: _missing(text, r.answers))
                for _ in range(n)
            ],
            return_exceptions=True,
        )
        agree = 0
        for r in reads:
            if isinstance(r, BaseException):
                continue
            if not qa["answerable"]:
                agree += not r.answerable
            elif r.answerable:
                pred = [(s["start"], s["end"]) for s in _spans_with_offsets(text, r.answers)]
                agree += overlap_coefficient(gold, pred) >= 0.8
        return agree / n


# =========================================================================== Extractive summarization


class SummaryAnswer(BaseModel):
    spans: list[str] = Field(description="Дословные фрагменты текста, в порядке появления")


class ExtractiveSummarization(Task[SummarizationSchema]):
    name = "extractive_summarization"
    schema_cls = SummarizationSchema
    proposal_hint = (
        "Каждая задача задаёт свой фокус выжимки (query), например «риски», «решения и сроки», «цифры».",
        "Each task sets its own summary focus (query), e.g. 'risks', 'decisions and deadlines', 'figures'.",
    )

    def answer_model(self, schema):
        return SummaryAnswer

    def instruction(self, schema, lang):
        focus = (
            L(lang, f"Выбирай только то, что относится к запросу: «{schema.query}».", f"Select only what is relevant to the query: '{schema.query}'.")
            if schema.query
            else L(lang, "Выбирай самое важное содержание текста.", "Select the most important content of the text.")
        )
        return L(
            lang,
            f"{schema.instruction}\nСделай экстрактивную выжимку: выпиши до {schema.max_spans} фрагментов текста "
            f"дословно (фрагмент может быть частью предложения). {focus} Не перефразируй и не повторяйся. "
            f"Если релевантного нет, верни пустой список.",
            f"{schema.instruction}\nMake an extractive summary: copy up to {schema.max_spans} text fragments verbatim "
            f"(a fragment may be part of a sentence). {focus} Do not paraphrase or repeat. "
            f"If nothing is relevant, return an empty list.",
        )

    def validate(self, text, schema, raw):
        spans = raw.spans  # type: ignore[attr-defined]
        errors = _missing(text, spans)
        if len(spans) > schema.max_spans:
            errors.append(f"Не больше {schema.max_spans} фрагментов, получено {len(spans)}")
        errors += [f"Фрагмент «{s}» слишком короткий" for s in spans if len(s.split()) < 3]
        return errors

    def repair(self, text, schema, raw):
        spans = [s for s in raw.spans if find_first(text, s) and len(s.split()) >= 3]  # type: ignore[attr-defined]
        return raw.model_copy(update={"spans": spans[: schema.max_spans]})

    def postprocess(self, text, schema, raw):
        spans = sorted(_spans_with_offsets(text, raw.spans), key=lambda s: s["start"])  # type: ignore[attr-defined]
        return {"spans": spans}

    def aggregate(self, text, schema, answers):
        if len(answers) == 1:
            return answers[0], {}
        sets = [[(s["start"], s["end"]) for s in a["spans"]] for a in answers]
        sim = [[char_jaccard(a, b) for b in sets] for a in sets]
        # Медоида: сэмпл, наиболее похожий на остальные; средняя похожесть = уверенность.
        scores = [(sum(row) - 1) / (len(row) - 1) for row in sim]
        best = max(range(len(answers)), key=scores.__getitem__)
        return answers[best], {"agreement": round(scores[best], 3)}

    def generation_hint(self, schema, target, lang):
        if schema.query:
            return L(lang, f"содержит как относящиеся к теме «{schema.query}» сведения, так и посторонние детали",
                     f"contains both information about '{schema.query}' and unrelated details")
        return super().generation_hint(schema, target, lang)


# =========================================================================== Tabular extraction

_PY_TYPES = {"int": int, "float": float, "bool": bool, "str": str, "date": str}


class TabularExtraction(Task[TabularSchema]):
    """Каждый документ превращается в строку таблицы по схеме колонок."""

    name = "tabular"
    schema_cls = TabularSchema
    proposal_hint = (
        "Каждая задача: таблица из 4-10 колонок разных типов (int/float/bool/category/str/date), значения которых "
        "можно извлечь из большинства текстов; укажи target — колонку, которую полезно предсказывать по остальным.",
        "Each task: a table of 4-10 columns of various types (int/float/bool/category/str/date) extractable from most "
        "texts; set target to a column worth predicting from the others.",
    )

    def answer_model(self, schema):
        fields: dict[str, Any] = {}
        for col in schema.columns:
            if col.dtype == "category" and col.categories:
                py = literal_of(col.categories)
            else:
                py = _PY_TYPES.get(col.dtype, str)
            fields[col.name] = (Optional[py], Field(default=None, description=col.description))
        return create_model("Row", **fields)

    def instruction(self, schema, lang):
        cols = []
        for c in schema.columns:
            extra = f"; {L(lang, 'значения', 'values')}: {', '.join(c.categories)}" if c.categories else ""
            unit = f"; {L(lang, 'единица', 'unit')}: {c.unit}" if c.unit else ""
            cols.append(f"- {c.name} ({c.dtype}{unit}{extra}): {c.description}")
        return L(
            lang,
            f"{schema.instruction}\nИзвлеки из текста значения колонок. Если значение не указано и не следует из "
            f"текста однозначно, ставь null — не выдумывай. Даты в формате YYYY-MM-DD.\nКолонки:\n" + "\n".join(cols),
            f"{schema.instruction}\nExtract column values from the text. If a value is not stated and does not follow "
            f"unambiguously, use null — do not guess. Dates as YYYY-MM-DD.\nColumns:\n" + "\n".join(cols),
        )

    def check_schema(self, schema):
        errors = [f"У категориальной колонки {c.name} нет categories" for c in schema.columns if c.dtype == "category" and not c.categories]
        if schema.target and schema.target not in {c.name for c in schema.columns}:
            errors.append(f"target {schema.target} отсутствует среди колонок")
        return errors

    def validate(self, text, schema, raw):
        errors = []
        for c in schema.columns:
            value = getattr(raw, c.name)
            if c.dtype == "date" and value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    errors.append(f"Колонка {c.name}: «{value}» не дата в формате YYYY-MM-DD")
        return errors

    def postprocess(self, text, schema, raw):
        return {"row": raw.model_dump()}

    def aggregate(self, text, schema, answers):
        n = len(answers)
        row: dict[str, Any] = {}
        conf: dict[str, float] = {}
        for c in schema.columns:
            values = [a["row"][c.name] for a in answers]
            if c.dtype in ("int", "float") and any(v is not None for v in values):
                nums = sorted(v for v in values if v is not None)
                if len(nums) / n < 0.5:
                    row[c.name] = None
                    conf[c.name] = round(1 - len(nums) / n, 3)
                    continue
                med = nums[len(nums) // 2]
                row[c.name] = med
                conf[c.name] = round(sum(v is not None and abs(v - med) <= 0.01 * max(abs(med), 1) for v in values) / n, 3)
            else:
                value, count = Counter(values).most_common(1)[0]
                row[c.name] = value
                conf[c.name] = round(count / n, 3)
        return {"row": row}, {"column_agreement": conf}

    def generation_hint(self, schema, target, lang):
        cols = ", ".join(c.name for c in schema.columns)
        return L(lang, f"описывает один объект, для которого можно определить: {cols}",
                 f"describes one object for which one can determine: {cols}")
