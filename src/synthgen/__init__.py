"""synthgen — генерация синтетических обучающих данных из корпуса текстов с помощью LLM."""

__version__ = "0.1.0"

from synthgen import export
from synthgen.config import LLMConfig
from synthgen.corpus import chunk_documents, detect_lang, from_texts, read_jsonl, read_txt_dir, select_diverse
from synthgen.dataset import Dataset
from synthgen.llm import LLMClient, LLMError
from synthgen.pipeline import GenerationConfig, Generator
from synthgen.quality import deduplicate, label_distribution
from synthgen.record import Document, Record
from synthgen.schemas import (
    ClassificationSchema,
    ColumnDef,
    LabelDef,
    NERSchema,
    QASchema,
    RegressionSchema,
    RelationDef,
    RelationSchema,
    SummarizationSchema,
    TabularSchema,
)
from synthgen.tasks import TASKS, Task, get_task, register_task

__all__ = [
    "__version__", "export", "LLMConfig", "LLMClient", "LLMError", "Document", "Record", "Dataset", "Generator", "GenerationConfig",
    "chunk_documents", "detect_lang", "from_texts", "read_jsonl", "read_txt_dir", "select_diverse",
    "deduplicate", "label_distribution", "TASKS", "Task", "get_task", "register_task",
    "ClassificationSchema", "RegressionSchema", "NERSchema", "RelationSchema", "QASchema", "SummarizationSchema",
    "TabularSchema", "LabelDef", "RelationDef", "ColumnDef",
]
