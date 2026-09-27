"""Подсветка спанов в Jupyter."""

from __future__ import annotations

import html
from typing import Any

_PALETTE = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fbcfe8", "#ddd6fe", "#fed7aa", "#99f6e4", "#e5e7eb"]


def highlight(text: str, spans: list[dict[str, Any]], label_key: str | None = "type") -> str:
    """HTML с подсвеченными непересекающимися спанами (для вложенных берётся внешний)."""
    colors: dict[str, str] = {}
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: (s["start"], -s["end"])):
        if s["start"] < pos:
            continue
        label = str(s.get(label_key, "")) if label_key else ""
        color = colors.setdefault(label, _PALETTE[len(colors) % len(_PALETTE)])
        out.append(html.escape(text[pos : s["start"]]))
        tag = f'<sup style="font-size:0.7em;font-weight:600;margin-left:2px">{html.escape(label)}</sup>' if label else ""
        out.append(f'<mark style="background:{color};color:#111;padding:1px 3px;border-radius:3px">'
                   f'{html.escape(text[s["start"]:s["end"]])}{tag}</mark>')
        pos = s["end"]
    out.append(html.escape(text[pos:]))
    return f'<div style="line-height:1.9;max-width:900px">{"".join(out)}</div>'
