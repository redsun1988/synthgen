from synthgen.tasks.base import Task
from synthgen.tasks.classification import TextClassification, TextRegression
from synthgen.tasks.extractive import ExtractiveQA, ExtractiveSummarization, TabularExtraction
from synthgen.tasks.ner import NER, RelationExtraction

TASKS: dict[str, type[Task]] = {
    t.name: t
    for t in (TextClassification, TextRegression, NER, RelationExtraction, ExtractiveQA, ExtractiveSummarization, TabularExtraction)
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Неизвестная задача {name!r}. Доступны: {sorted(TASKS)}")
    return TASKS[name]()


def register_task(task_cls: type[Task]) -> type[Task]:
    """Регистрирует пользовательскую задачу (можно как декоратор), чтобы её находили Generator и экспортёры."""
    TASKS[task_cls.name] = task_cls
    return task_cls


__all__ = [
    "TASKS",
    "Task",
    "get_task",
    "register_task",
    "TextClassification",
    "TextRegression",
    "NER",
    "RelationExtraction",
    "ExtractiveQA",
    "ExtractiveSummarization",
    "TabularExtraction",
]
