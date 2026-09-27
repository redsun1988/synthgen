from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from synthgen.config import LLMConfig

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
Messages = list[dict[str, str]]
# Возвращает список ошибок; пустой список означает, что ответ корректен.
Validator = Callable[[Any], list[str]]


class LLMError(RuntimeError):
    pass


@dataclass
class UsageStats:
    calls: int = 0
    failed_calls: int = 0
    retries: int = 0
    repaired: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def inline_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Раскрывает $ref/$defs: не все OpenAI-совместимые серверы понимают ссылки в JSON Schema."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(raw: str) -> Any:
    raw = _THINK_RE.sub("", raw).strip()
    raw = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def _compact_schema(schema: Any) -> Any:
    """Схема без служебных полей (title), чтобы подсказка в промпте была короче."""
    if isinstance(schema, dict):
        return {k: _compact_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_compact_schema(v) for v in schema]
    return schema


def _with_schema_hint(messages: Messages, schema: dict[str, Any]) -> Messages:
    hint = "\n\nФормат ответа: JSON по схеме (description — смысл поля):\n" + json.dumps(
        _compact_schema(schema), ensure_ascii=False, separators=(",", ":")
    )
    last = len(messages) - 1 - next(i for i, m in enumerate(reversed(messages)) if m["role"] == "user")
    return [*messages[:last], {**messages[last], "content": messages[last]["content"] + hint}, *messages[last + 1 :]]


class LLMClient:
    """Асинхронный клиент OpenAI-совместимого API со структурированным выводом.

    - ответ валидируется Pydantic-моделью и пользовательским валидатором;
    - при ошибке модель получает текст ошибки и исправляет ответ (до `max_retries` раз);
    - число одновременных запросов ограничено `max_concurrency`.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self.usage = UsageStats()
        self._client: AsyncOpenAI | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._embed_client: AsyncOpenAI | None = None
        self._embed_semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop_bound(self) -> None:
        # AsyncOpenAI и Semaphore привязаны к event loop; пересоздаём их, если loop сменился.
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            cfg = self.config
            self._loop = loop
            self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout, max_retries=2)
            self._semaphore = asyncio.Semaphore(cfg.max_concurrency)
            if cfg.separate_embed_server:
                # Свой клиент и свой лимит: эмбеддинги не занимают слоты чат-модели.
                self._embed_client = AsyncOpenAI(
                    base_url=cfg.embed_base_url or cfg.base_url,
                    api_key=cfg.embed_api_key or cfg.api_key,
                    timeout=cfg.timeout,
                    max_retries=2,
                )
                self._embed_semaphore = asyncio.Semaphore(cfg.max_concurrency)
            else:
                self._embed_client, self._embed_semaphore = self._client, self._semaphore

    def _request_params(self, temperature: float, max_tokens: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.config.model, "temperature": temperature}
        if max_tokens:
            params["max_tokens"] = max_tokens
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort
        params.update(self.config.extra_params)
        return params

    async def _complete(self, messages: Messages, **params: Any) -> str:
        self._ensure_loop_bound()
        assert self._client and self._semaphore
        async with self._semaphore:
            self.usage.calls += 1
            try:
                response = await self._client.chat.completions.create(messages=messages, **params)
            except Exception:
                self.usage.failed_calls += 1
                raise
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens or 0
            self.usage.completion_tokens += response.usage.completion_tokens or 0
        return response.choices[0].message.content or ""

    async def text(
        self, messages: Messages, temperature: float = 0.7, max_tokens: int | None = None
    ) -> str:
        raw = await self._complete(messages, **self._request_params(temperature, max_tokens))
        return _THINK_RE.sub("", raw).strip()

    async def structured(
        self,
        messages: Messages,
        response_model: type[M],
        temperature: float = 0.2,
        validator: Validator | None = None,
        max_tokens: int | None = None,
        repair: Callable[[M], M | None] | None = None,
        schema_in_prompt: bool = True,
    ) -> M:
        """Запрашивает ответ по JSON-схеме `response_model` с циклом самоисправления.

        Если после всех попыток ответ всё ещё не проходит `validator`, но синтаксически корректен,
        `repair` может отбросить невалидные части (например, спаны, которых нет в тексте).

        `schema_in_prompt`: многие серверы (Ollama, llama.cpp) используют JSON Schema только для ограничения
        грамматики, и модель не видит описаний полей. Поэтому по умолчанию компактная схема дописывается в промпт.
        """
        schema = inline_json_schema(response_model.model_json_schema())
        if schema_in_prompt:
            messages = _with_schema_hint(messages, schema)
        params = self._request_params(temperature, max_tokens)
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": response_model.__name__, "schema": schema},
        }
        history = list(messages)
        last_error = ""
        last_parsed: M | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                self.usage.retries += 1
            raw = await self._complete(history, **params)
            try:
                parsed = response_model.model_validate(extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = f"Ответ не соответствует JSON-схеме: {str(e)[:1500]}"
            else:
                last_parsed = parsed
                errors = validator(parsed) if validator else []
                if not errors:
                    return parsed
                last_error = "Ответ содержит ошибки:\n- " + "\n- ".join(errors[:20])
            logger.debug("Попытка %d отклонена: %s", attempt + 1, last_error)
            history = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"{last_error}\nИсправь ответ и верни только корректный JSON."},
            ]
        if repair is not None and last_parsed is not None:
            fixed = repair(last_parsed)
            if fixed is not None:
                self.usage.repaired += 1
                return fixed
        raise LLMError(f"Не удалось получить валидный ответ за {self.config.max_retries + 1} попыток: {last_error}")

    async def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not self.config.embed_model:
            raise LLMError("Не задана модель эмбеддингов (LLMConfig.embed_model)")
        self._ensure_loop_bound()
        assert self._embed_client and self._embed_semaphore
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            async with self._embed_semaphore:
                response = await self._embed_client.embeddings.create(
                    model=self.config.embed_model, input=texts[i : i + batch_size]
                )
            vectors.extend(d.embedding for d in response.data)
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
