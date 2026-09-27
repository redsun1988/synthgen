from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field, create_model

from synthgen.schemas import ClassificationSchema, RegressionSchema
from synthgen.tasks.base import L, Lang, Task, literal_of


class TextClassification(Task[ClassificationSchema]):
    name = "text_classification"
    schema_cls = ClassificationSchema
    proposal_hint = (
        "Каждая задача: 2-6 взаимоисключающих классов с чёткими описаниями, классы должны встречаться в корпусе.",
        "Each task: 2-6 mutually exclusive classes with crisp descriptions; classes must occur in the corpus.",
    )

    def answer_model(self, schema: ClassificationSchema) -> type[BaseModel]:
        labels = literal_of([lab.name for lab in schema.labels])
        if schema.multi_label:
            return create_model("MultiLabelAnswer", labels=(list[labels], Field(description="Все подходящие классы")))
        return create_model("ClassificationAnswer", label=(labels, ...))

    def instruction(self, schema: ClassificationSchema, lang: Lang) -> str:
        classes = "\n".join(f"- {lab.name}: {lab.description}" for lab in schema.labels)
        mode = (
            L(lang, "Выбери все подходящие классы (можно ни одного).", "Choose all applicable classes (possibly none).")
            if schema.multi_label
            else L(lang, "Выбери ровно один класс.", "Choose exactly one class.")
        )
        return f"{schema.instruction}\n{mode}\n{L(lang, 'Классы', 'Classes')}:\n{classes}"

    def postprocess(self, text: str, schema: ClassificationSchema, raw: BaseModel) -> dict[str, Any]:
        if schema.multi_label:
            return {"labels": sorted(set(raw.labels))}  # type: ignore[attr-defined]
        return {"label": raw.label}  # type: ignore[attr-defined]

    def aggregate(self, text, schema, answers):
        n = len(answers)
        names = [lab.name for lab in schema.labels]
        if schema.multi_label:
            probs = {name: sum(name in a["labels"] for a in answers) / n for name in names}
            return {"labels": [k for k, p in probs.items() if p >= 0.5]}, {"label_probs": probs}
        counts = Counter(a["label"] for a in answers)
        probs = {name: counts.get(name, 0) / n for name in names}
        return {"label": counts.most_common(1)[0][0]}, {"label_probs": probs}

    def generation_targets(self, schema):
        return [lab.name for lab in schema.labels]

    def generation_hint(self, schema, target, lang):
        desc = next(lab.description for lab in schema.labels if lab.name == target)
        return L(
            lang,
            f"однозначно относится к классу «{target}» ({desc}) в задаче «{schema.instruction}»",
            f"unambiguously belongs to class '{target}' ({desc}) in the task '{schema.instruction}'",
        )

    def target_matches(self, schema, target, answer):
        if schema.multi_label:
            return target in answer["labels"]
        return answer["label"] == target


class TextRegression(Task[RegressionSchema]):
    name = "text_regression"
    schema_cls = RegressionSchema
    proposal_hint = (
        "Каждая задача: одна числовая характеристика текста с понятной шкалой (например, срочность 0-10).",
        "Each task: one numeric property of the text with a clear scale (e.g. urgency 0-10).",
    )

    def answer_model(self, schema):
        return create_model("RegressionAnswer", value=(float, Field(ge=schema.min_value, le=schema.max_value)))

    def instruction(self, schema, lang):
        return L(
            lang,
            f"{schema.instruction}\nОцени величину «{schema.target}» числом от {schema.min_value} до {schema.max_value}.\n"
            f"Шкала: {schema.scale_description}",
            f"{schema.instruction}\nRate '{schema.target}' as a number from {schema.min_value} to {schema.max_value}.\n"
            f"Scale: {schema.scale_description}",
        )

    def check_schema(self, schema):
        return [] if schema.max_value > schema.min_value else ["max_value должен быть больше min_value"]

    def postprocess(self, text, schema, raw):
        return {"value": float(raw.value)}  # type: ignore[attr-defined]

    def aggregate(self, text, schema, answers):
        values = sorted(a["value"] for a in answers)
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        median = values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
        return {"value": median}, {"mean": mean, "std": std, "samples": values}

    def generation_targets(self, schema):
        # 5 равномерных точек шкалы, чтобы таргет покрывал весь диапазон.
        step = (schema.max_value - schema.min_value) / 4
        return [round(schema.min_value + i * step, 2) for i in range(5)]

    def generation_hint(self, schema, target, lang):
        return L(
            lang,
            f"имеет значение «{schema.target}» примерно {target} по шкале от {schema.min_value} до {schema.max_value} ({schema.scale_description})",
            f"has '{schema.target}' of about {target} on a scale from {schema.min_value} to {schema.max_value} ({schema.scale_description})",
        )

    def target_matches(self, schema, target, answer):
        return abs(answer["value"] - target) <= 0.2 * (schema.max_value - schema.min_value)
