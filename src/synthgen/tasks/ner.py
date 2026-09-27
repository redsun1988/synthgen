from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Optional

from pydantic import BaseModel, Field, create_model

from synthgen.schemas import NERSchema, RelationSchema
from synthgen.spans import find_all, overlaps
from synthgen.tasks.base import L, Lang, Task, literal_of

VOTE_THRESHOLD = 0.5


def _mention_model(schema: NERSchema) -> type[BaseModel]:
    fields: dict[str, Any] = {
        "text": (str, Field(description="Фрагмент, скопированный из текста дословно")),
        "type": (literal_of([t.name for t in schema.entity_types]), ...),
    }
    if schema.with_attributes:
        fields["value"] = (Optional[str], Field(default=None, description="Нормализованное значение"))
    return create_model("Mention", **fields)


def _types_block(schema: NERSchema) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in schema.entity_types)


def _entity_instruction(schema: NERSchema, lang: Lang) -> str:
    parts = [
        schema.instruction,
        L(lang, "Типы сущностей:", "Entity types:"),
        _types_block(schema),
        L(
            lang,
            "Поле text копируй из текста дословно, без изменения формы слов. Каждое упоминание указывай один раз. "
            "Спан — только само имя: без должностей, титулов и родовых слов (не «губернатор Иван Петров», а «Иван Петров»; "
            "не «компания «Яндекс»», а «Яндекс»).",
            "Copy the text field verbatim from the text, without changing word forms. List each mention once. "
            "A span is the name only: no job titles or generic words (not 'CEO John Smith' but 'John Smith').",
        ),
    ]
    if schema.allow_nested:
        parts.append(L(lang, "Вложенные сущности разрешены (например, город внутри названия организации).",
                       "Nested entities are allowed (e.g. a city inside an organization name)."))
    if schema.with_attributes:
        parts.append(L(lang, "В value укажи нормализованное значение: даты в ISO 8601, суммы числом с валютой, имена в именительном падеже.",
                       "Put a normalized value into value: dates in ISO 8601, amounts as number with currency, names in canonical form."))
    return "\n".join(parts)


def _locate_mentions(text: str, mentions: list[BaseModel]) -> list[str]:
    return [f"Фрагмент «{m.text}» не найден в тексте дословно" for m in mentions if not find_all(text, m.text)]  # type: ignore[attr-defined]


def _to_entities(text: str, schema: NERSchema, mentions: list[BaseModel]) -> list[dict[str, Any]]:
    entities: dict[tuple[int, int, str], dict[str, Any]] = {}
    for m in mentions:
        for start, end in find_all(text, m.text):  # type: ignore[attr-defined]
            ent = {"start": start, "end": end, "text": text[start:end], "type": m.type}  # type: ignore[attr-defined]
            if schema.with_attributes:
                ent["value"] = getattr(m, "value", None)
            entities.setdefault((start, end, ent["type"]), ent)
    result = sorted(entities.values(), key=lambda e: (e["start"], -e["end"]))
    return result if schema.allow_nested else _drop_overlaps(result)


def _drop_overlaps(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ent in sorted(entities, key=lambda e: e["end"] - e["start"], reverse=True):
        if not any(overlaps((ent["start"], ent["end"]), (k["start"], k["end"])) for k in kept):
            kept.append(ent)
    return sorted(kept, key=lambda e: e["start"])


def _vote_entities(schema: NERSchema, answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(answers)
    votes: Counter[tuple[int, int, str]] = Counter()
    samples: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for a in answers:
        for e in a["entities"]:
            key = (e["start"], e["end"], e["type"])
            votes[key] += 1
            samples[key].append(e)
    result = []
    for key, count in votes.items():
        if count / n < VOTE_THRESHOLD:
            continue
        ent = dict(samples[key][0])
        if schema.with_attributes:
            ent["value"] = Counter(s.get("value") for s in samples[key]).most_common(1)[0][0]
        ent["confidence"] = round(count / n, 3)
        result.append(ent)
    result.sort(key=lambda e: (e["start"], -e["end"]))
    return result if schema.allow_nested else _drop_overlaps(result)


class NER(Task[NERSchema]):
    name = "ner"
    schema_cls = NERSchema
    proposal_hint = (
        "Каждая задача: 2-6 типов сущностей, которые реально встречаются в этих текстах (не только PER/ORG/LOC, "
        "но и доменные: препараты, суммы, продукты, симптомы и т.п.).",
        "Each task: 2-6 entity types that actually occur in these texts (not only PER/ORG/LOC but also domain "
        "specific: drugs, amounts, products, symptoms, etc.).",
    )

    def answer_model(self, schema):
        return create_model("NERAnswer", entities=(list[_mention_model(schema)], ...))

    def instruction(self, schema, lang):
        return _entity_instruction(schema, lang)

    def validate(self, text, schema, raw):
        return _locate_mentions(text, raw.entities)  # type: ignore[attr-defined]

    def repair(self, text, schema, raw):
        return raw.model_copy(update={"entities": [m for m in raw.entities if find_all(text, m.text)]})  # type: ignore[attr-defined]

    def postprocess(self, text, schema, raw):
        return {"entities": _to_entities(text, schema, raw.entities)}  # type: ignore[attr-defined]

    def aggregate(self, text, schema, answers):
        entities = _vote_entities(schema, answers)
        return {"entities": entities}, {"entity_confidence": [e["confidence"] for e in entities], "n_samples": len(answers)}

    def generation_hint(self, schema, target, lang):
        types = ", ".join(t.name for t in schema.entity_types)
        return L(lang, f"содержит несколько упоминаний сущностей типов: {types}",
                 f"contains several mentions of entities of types: {types}")


class RelationExtraction(Task[RelationSchema]):
    name = "relation_extraction"
    schema_cls = RelationSchema
    proposal_hint = (
        "Каждая задача: 2-5 типов сущностей и 1-4 направленных типа отношений между ними, которые явно выражены в текстах.",
        "Each task: 2-5 entity types and 1-4 directed relation types between them that are explicitly stated in the texts.",
    )

    def answer_model(self, schema):
        relation = create_model(
            "Relation",
            head=(str, Field(description="text сущности-источника из списка entities")),
            relation=(literal_of([r.name for r in schema.relation_types]), ...),
            tail=(str, Field(description="text сущности-цели из списка entities")),
        )
        return create_model("RelationAnswer", entities=(list[_mention_model(schema)], ...), relations=(list[relation], ...))

    def instruction(self, schema, lang):
        rels = "\n".join(
            f"- {r.name}: {r.description} ({'/'.join(r.head_types)} -> {'/'.join(r.tail_types)})" for r in schema.relation_types
        )
        return (
            f"{_entity_instruction(schema, lang)}\n"
            + L(lang, "Затем выдели отношения, явно выраженные в тексте. Типы отношений:",
                "Then extract relations explicitly stated in the text. Relation types:")
            + f"\n{rels}\n"
            + L(lang, "head и tail должны совпадать с полем text одной из выделенных сущностей.",
                "head and tail must equal the text field of one of the extracted entities.")
        )

    def check_schema(self, schema):
        types = {t.name for t in schema.entity_types}
        return [
            f"Отношение {r.name} ссылается на неизвестный тип сущности {t}"
            for r in schema.relation_types
            for t in [*r.head_types, *r.tail_types]
            if t not in types
        ]

    def validate(self, text, schema, raw):
        errors = _locate_mentions(text, raw.entities)  # type: ignore[attr-defined]
        by_text = {m.text: m.type for m in raw.entities}  # type: ignore[attr-defined]
        rel_defs = {r.name: r for r in schema.relation_types}
        for r in raw.relations:  # type: ignore[attr-defined]
            for role, value in (("head", r.head), ("tail", r.tail)):
                if value not in by_text:
                    errors.append(f"{role} «{value}» отношения {r.relation} отсутствует в entities")
            rd = rel_defs[r.relation]
            if r.head in by_text and by_text[r.head] not in rd.head_types:
                errors.append(f"У отношения {r.relation} head должен иметь тип {rd.head_types}, а «{r.head}» имеет {by_text[r.head]}")
            if r.tail in by_text and by_text[r.tail] not in rd.tail_types:
                errors.append(f"У отношения {r.relation} tail должен иметь тип {rd.tail_types}, а «{r.tail}» имеет {by_text[r.tail]}")
        return errors

    def repair(self, text, schema, raw):
        entities = [m for m in raw.entities if find_all(text, m.text)]  # type: ignore[attr-defined]
        by_text = {m.text: m.type for m in entities}
        rel_defs = {r.name: r for r in schema.relation_types}
        relations = [
            r for r in raw.relations  # type: ignore[attr-defined]
            if by_text.get(r.head) in rel_defs[r.relation].head_types and by_text.get(r.tail) in rel_defs[r.relation].tail_types
        ]
        return raw.model_copy(update={"entities": entities, "relations": relations})

    def postprocess(self, text, schema, raw):
        entities = _to_entities(text, schema, raw.entities)  # type: ignore[attr-defined]
        relations = sorted({(r.head, r.relation, r.tail) for r in raw.relations if r.head != r.tail})  # type: ignore[attr-defined]
        return {"entities": entities, "relations": [{"head": h, "type": t, "tail": tl} for h, t, tl in relations]}

    def aggregate(self, text, schema, answers):
        n = len(answers)
        entities = _vote_entities(schema, answers)
        first_by_text: dict[str, int] = {}
        for i, e in enumerate(entities):
            first_by_text.setdefault(e["text"].lower(), i)
        votes = Counter((r["head"], r["type"], r["tail"]) for a in answers for r in a["relations"])
        relations = []
        for (head, rtype, tail), count in votes.items():
            h, t = first_by_text.get(head.lower()), first_by_text.get(tail.lower())
            if count / n >= VOTE_THRESHOLD and h is not None and t is not None:
                relations.append({"head": h, "tail": t, "type": rtype, "head_text": head, "tail_text": tail,
                                  "confidence": round(count / n, 3)})
        return {"entities": entities, "relations": relations}, {"n_samples": n}

    def generation_hint(self, schema, target, lang):
        rels = ", ".join(r.name for r in schema.relation_types)
        return L(lang, f"явно описывает связи типов {rels} между конкретными сущностями",
                 f"explicitly describes relations of types {rels} between concrete entities")
