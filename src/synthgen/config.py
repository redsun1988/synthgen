from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Настройки OpenAI-совместимого эндпоинта (Ollama, vLLM, OpenAI и т.п.)."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3:8b"
    embed_model: str | None = None
    # Отдельный сервер для эмбеддингов; None — использовать base_url / api_key основного сервера.
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    max_concurrency: int = 4
    timeout: float = 600.0
    max_retries: int = 3
    # Для thinking-моделей: "none" отключает встроенный ризонинг и ускоряет генерацию в разы.
    reasoning_effort: str | None = "none"
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None, **overrides: Any) -> "LLMConfig":
        """Читает переменные SYNTHGEN_* из окружения и `.env` (по умолчанию ищется от текущей папки вверх)."""
        load_dotenv(dotenv_path)
        env = {
            "base_url": os.getenv("SYNTHGEN_BASE_URL"),
            "api_key": os.getenv("SYNTHGEN_API_KEY"),
            "model": os.getenv("SYNTHGEN_MODEL"),
            "embed_model": os.getenv("SYNTHGEN_EMBED_MODEL"),
            "embed_base_url": os.getenv("SYNTHGEN_EMBED_BASE_URL"),
            "embed_api_key": os.getenv("SYNTHGEN_EMBED_API_KEY"),
            "max_concurrency": os.getenv("SYNTHGEN_MAX_CONCURRENCY"),
            "reasoning_effort": os.getenv("SYNTHGEN_REASONING_EFFORT"),
        }
        kwargs: dict[str, Any] = {k: v for k, v in env.items() if v not in (None, "")}
        if "max_concurrency" in kwargs:
            kwargs["max_concurrency"] = int(kwargs["max_concurrency"])
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def separate_embed_server(self) -> bool:
        return bool(self.embed_base_url or self.embed_api_key)
