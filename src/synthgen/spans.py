"""Привязка спанов к тексту.

LLM плохо считает символьные смещения, поэтому модель возвращает сам фрагмент текста,
а смещения находит фреймворк. Фрагмент, которого нет в тексте, считается ошибкой валидации.
"""

from __future__ import annotations

import re

_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "’": "'", "‘": "'", "—": "-", "–": "-"})


def _pattern(fragment: str) -> re.Pattern[str]:
    tokens = fragment.translate(_QUOTES).split()
    body = r"\s+".join(re.escape(t) for t in tokens)
    return re.compile(body, re.IGNORECASE)


def _stem_pattern(fragment: str) -> re.Pattern[str]:
    """Каждое слово длиннее 4 символов заменяется основой + окончанием: «Набережные Челны» -> «Набережных Челнах»."""
    parts = []
    for tok in fragment.translate(_QUOTES).split():
        if len(tok) > 4 and tok[-1].isalpha():
            parts.append(re.escape(tok[: max(4, len(tok) - 3)]) + r"\w*")
        else:
            parts.append(re.escape(tok))
    return re.compile(r"(?<!\w)" + r"\s+".join(parts), re.IGNORECASE)


def _is_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() and text[start].isalnum()) and not (after.isalnum() and text[end - 1].isalnum())


def find_all(text: str, fragment: str, word_boundary: bool = True, morph: bool = True) -> list[tuple[int, int]]:
    """Все вхождения фрагмента. Порядок попыток: точный поиск; без учёта регистра, кавычек и пробелов;
    по основам слов (другая словоформа, важно для русского)."""
    fragment = fragment.strip()
    if not fragment:
        return []
    hits: list[tuple[int, int]] = []
    start = text.find(fragment)
    while start != -1:
        hits.append((start, start + len(fragment)))
        start = text.find(fragment, start + 1)
    normalized = text.translate(_QUOTES)  # translate сохраняет длину, смещения совпадают
    if not hits:
        hits = [(m.start(), m.end()) for m in _pattern(fragment).finditer(normalized)]
    if word_boundary:
        hits = [(s, e) for s, e in hits if _is_boundary(text, s, e)]
    if not hits and morph:
        hits = [(m.start(), m.end()) for m in _stem_pattern(fragment).finditer(normalized)]
    return hits


def find_first(text: str, fragment: str) -> tuple[int, int] | None:
    hits = find_all(text, fragment) or find_all(text, fragment, word_boundary=False, morph=False)
    return hits[0] if hits else None


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def char_jaccard(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Сходство двух наборов спанов по покрытым символам."""
    sa = {i for s, e in a for i in range(s, e)}
    sb = {i for s, e in b for i in range(s, e)}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def overlap_coefficient(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """|A∩B| / min(|A|, |B|): 1.0, если один ответ содержит другой (короткий ответ внутри цитаты предложения)."""
    sa = {i for s, e in a for i in range(s, e)}
    sb = {i for s, e in b for i in range(s, e)}
    if not sa or not sb:
        return float(sa == sb)
    return len(sa & sb) / min(len(sa), len(sb))
