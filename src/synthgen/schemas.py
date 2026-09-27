"""Схемы задач. Схема описывает, ЧТО нужно разметить, и может быть задана вручную или придумана LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LabelDef(BaseModel):
    name: str = Field(description="Короткое имя метки в snake_case, 1-3 слова: complaint, positive, high_risk")
    description: str = Field(description="Однозначный критерий, по которому текст относят к метке")


class TaskSchema(BaseModel):
    name: str = Field(description="Уникальное имя задачи в snake_case, отражающее её смысл: review_sentiment, request_urgency")
    instruction: str = Field(description="Что именно нужно определить в тексте, 1-2 предложения")


class ClassificationSchema(TaskSchema):
    labels: list[LabelDef] = Field(
        min_length=2,
        description="Общие категории, применимые к многим текстам такого рода (не пересказ конкретного текста); "
        "взаимоисключающие, если multi_label=false",
    )
    multi_label: bool = False


class RegressionSchema(TaskSchema):
    target: str = Field(description="Название оцениваемой величины, 1-3 слова: срочность, удовлетворённость клиента")
    min_value: float = Field(description="Нижняя граница шкалы, например 0")
    max_value: float = Field(description="Верхняя граница шкалы, например 10")
    scale_description: str = Field(description="Что означают минимум, середина и максимум шкалы")


class NERSchema(TaskSchema):
    entity_types: list[LabelDef] = Field(min_length=1)
    allow_nested: bool = Field(default=False, description="Разрешены вложенные/перекрывающиеся сущности")
    with_attributes: bool = Field(
        default=False, description="Для каждой сущности вернуть нормализованное значение (дата в ISO, сумма числом)"
    )


class RelationDef(BaseModel):
    name: str
    description: str
    head_types: list[str] = Field(description="Допустимые типы сущности-источника")
    tail_types: list[str] = Field(description="Допустимые типы сущности-цели")


class RelationSchema(NERSchema):
    relation_types: list[RelationDef] = Field(min_length=1)


class QASchema(TaskSchema):
    n_questions: int = Field(default=3, ge=1, le=10)
    unanswerable_ratio: float = Field(default=0.3, ge=0, le=1, description="Доля вопросов, ответа на которые нет в тексте")
    allow_multi_span: bool = True
    question_focus: str | None = Field(default=None, description="Тематика вопросов, например 'сроки и суммы'")


class SummarizationSchema(TaskSchema):
    query: str | None = Field(default=None, description="Фокус выжимки; None означает общую выжимку")
    max_spans: int = Field(default=4, ge=1, le=15)


ColumnType = Literal["int", "float", "bool", "category", "str", "date"]


class ColumnDef(BaseModel):
    name: str = Field(description="Имя колонки в snake_case")
    dtype: ColumnType
    description: str
    categories: list[str] | None = Field(default=None, description="Допустимые значения для dtype=category")
    unit: str | None = None


class TabularSchema(TaskSchema):
    columns: list[ColumnDef] = Field(min_length=1)
    target: str | None = Field(default=None, description="Колонка-таргет для будущей модели")
