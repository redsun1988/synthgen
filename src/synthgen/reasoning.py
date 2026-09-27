"""Рассуждение постфактум: по тексту и уже проверенному ответу LLM пишет цепочку шагов, ведущую к ответу.

Ответ фиксирован до генерации рассуждения, поэтому рассуждение не может «увести» разметку не туда,
а в SFT-данные попадают только согласованные пары (рассуждение, ответ).
"""

from __future__ import annotations

import json
from typing import Any

from synthgen.llm import LLMClient
from synthgen.tasks.base import L, Lang


async def explain(llm: LLMClient, instruction: str, text: str, answer: dict[str, Any], lang: Lang) -> str:
    answer_json = json.dumps(answer, ensure_ascii=False)
    prompt = L(
        lang,
        "Ниже задание, текст и правильный ответ. Напиши рассуждение от первого лица, которое эксперт провёл бы, "
        "решая задание с нуля: какие фрагменты текста важны, какие варианты он рассматривает и почему отвергает, "
        "как приходит к ответу. Не упоминай, что ответ был известен заранее. 3-8 предложений, без JSON и без "
        f"финального ответа.\n\nЗАДАНИЕ:\n{instruction}\n\nТЕКСТ:\n\"\"\"\n{text}\n\"\"\"\n\nПРАВИЛЬНЫЙ ОТВЕТ:\n{answer_json}",
        "Below are a task, a text and the correct answer. Write the first-person reasoning an expert would do "
        "solving the task from scratch: which text fragments matter, which options they consider and reject, and how "
        "they reach the answer. Do not mention that the answer was known. 3-8 sentences, no JSON, no final answer."
        f"\n\nTASK:\n{instruction}\n\nTEXT:\n\"\"\"\n{text}\n\"\"\"\n\nCORRECT ANSWER:\n{answer_json}",
    )
    return await llm.text([{"role": "user", "content": prompt}], temperature=0.5)
