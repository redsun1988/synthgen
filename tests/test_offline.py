"""Тесты без реальной LLM: спаны, валидаторы, агрегация, экспорт, пайплайн на фейковом клиенте."""

import asyncio
import json

from synthgen import (
    ClassificationSchema, Dataset, Document, GenerationConfig, Generator, LabelDef, LLMClient, LLMConfig, NERSchema,
    QASchema, Record, TabularSchema, ColumnDef, export,
)
from synthgen.quality import near_duplicates_minhash
from synthgen.spans import find_all, find_first
from synthgen.tasks import NER, ExtractiveQA, TabularExtraction, TextClassification

TEXT = "Герман Греф заявил в Москве, что «Сбербанк» откроет офис в Москве и Казани."
NER_SCHEMA = NERSchema(
    name="ner", instruction="Найди сущности",
    entity_types=[LabelDef(name="PER", description="люди"), LabelDef(name="LOC", description="места"),
                  LabelDef(name="ORG", description="организации")],
)


def test_find_all_exact_normalized_and_boundaries():
    assert find_all(TEXT, "Москве") == [(21, 27), (59, 65)]
    assert find_first(TEXT, '"Сбербанк"') == (33, 43)  # кавычки-ёлочки нормализуются
    assert find_first(TEXT, "герман  греф") == (0, 11)
    assert find_all("Россия и Рос", "Рос") == [(9, 12)]  # не матчится внутри слова


def test_ner_validate_postprocess_vote():
    task = NER()
    model = task.answer_model(NER_SCHEMA)
    raw = model.model_validate({"entities": [{"text": "Герман Греф", "type": "PER"}, {"text": "Москве", "type": "LOC"},
                                             {"text": "Самаре", "type": "LOC"}]})
    assert task.validate(TEXT, NER_SCHEMA, raw) == ["Фрагмент «Самаре» не найден в тексте дословно"]
    a1 = task.postprocess(TEXT, NER_SCHEMA, raw)
    assert [e["text"] for e in a1["entities"]] == ["Герман Греф", "Москве", "Москве"]
    a2 = {"entities": a1["entities"][:1]}
    answer, _ = task.aggregate(TEXT, NER_SCHEMA, [a1, a2, a2])
    assert [e["text"] for e in answer["entities"]] == ["Герман Греф"]
    assert answer["entities"][0]["confidence"] == 1.0


def test_classification_soft_labels():
    schema = ClassificationSchema(name="s", instruction="x", labels=[LabelDef(name="a", description=""), LabelDef(name="b", description="")])
    answer, conf = TextClassification().aggregate("", schema, [{"label": "a"}, {"label": "a"}, {"label": "b"}, {"label": "a"}])
    assert answer == {"label": "a"} and conf["label_probs"] == {"a": 0.75, "b": 0.25}


def test_qa_validation():
    schema = QASchema(name="qa", instruction="Вопросы", n_questions=2, unanswerable_ratio=0.5)
    raw = ExtractiveQA().answer_model(schema).model_validate({"items": [
        {"question": "Кто?", "answerable": True, "answers": ["Герман Греф"]},
        {"question": "Когда?", "answerable": True, "answers": ["вчера"]},
    ]})
    errors = ExtractiveQA().validate(TEXT, schema, raw)
    assert any("без ответа" in e for e in errors) and any("вчера" in e for e in errors)


def test_tabular_aggregate_and_frame():
    schema = TabularSchema(name="t", instruction="x", columns=[
        ColumnDef(name="price", dtype="float", description=""),
        ColumnDef(name="city", dtype="category", description="", categories=["Москва", "Казань"]),
    ])
    task = TabularExtraction()
    answer, conf = task.aggregate("", schema, [{"row": {"price": 9.2, "city": "Казань"}}, {"row": {"price": 9.2, "city": "Казань"}},
                                               {"row": {"price": None, "city": "Москва"}}])
    assert answer["row"] == {"price": 9.2, "city": "Казань"}
    rec = Record(id="1", task="tabular", lang="ru", text="", task_schema=schema.model_dump(), answer=answer, doc_id="d")
    df = export.to_table([rec])
    assert str(df["city"].dtype) == "category" and df.loc[0, "price"] == 9.2


def test_bio_and_minhash_and_split():
    rec = Record(id="1", task="ner", lang="ru", text=TEXT, task_schema=NER_SCHEMA.model_dump(), doc_id="d",
                 answer={"entities": [{"start": 0, "end": 11, "text": "Герман Греф", "type": "PER"}]})
    assert export.to_bio(rec)[:3] == [("Герман", "B-PER"), ("Греф", "I-PER"), ("заявил", "O")]
    dup = near_duplicates_minhash([TEXT, TEXT.replace("Казани", "Самары"), "Совсем другой текст про погоду в горах"])
    assert dup == {1}
    ds = Dataset([rec.model_copy(update={"id": str(i), "doc_id": f"doc{i // 2}#{i % 2}"}) for i in range(20)])
    parts = ds.split(val=0.2, test=0.2)
    groups = {k: {r.doc_id.split("#")[0] for r in v} for k, v in parts.items()}
    assert not (groups["train"] & groups["test"]) and sum(map(len, parts.values())) == 20


class FakeLLM(LLMClient):
    """Отвечает первой меткой и считает вызовы; первый ответ намеренно ломаный, чтобы проверить самоисправление."""

    def __init__(self):
        super().__init__(LLMConfig(max_concurrency=2))
        self.broken_once = False

    async def _complete(self, messages, **params):
        self.usage.calls += 1
        if not self.broken_once:
            self.broken_once = True
            return "не JSON"
        name = params["response_format"]["json_schema"]["name"]
        assert name == "ClassificationAnswer"
        return json.dumps({"label": "pos"})


def test_pipeline_with_fake_llm_and_resume(tmp_path):
    schema = ClassificationSchema(name="s", instruction="Тональность",
                                  labels=[LabelDef(name="pos", description=""), LabelDef(name="neg", description="")])
    docs = [Document(id=f"d{i}", text=f"Текст номер {i}") for i in range(5)]
    out = tmp_path / "out.jsonl"
    llm = FakeLLM()
    ds = asyncio.run(Generator(llm, "text_classification", schema, n_samples=3, output_path=out).run(docs, progress=False))
    assert len(ds) == 5 and ds[0].confidence["label_probs"]["pos"] == 1.0
    assert llm.usage.retries == 1
    calls = llm.usage.calls
    gen = Generator(llm, "text_classification", schema, n_samples=3, output_path=out)
    ds2 = asyncio.run(gen.run(docs, progress=False))
    assert len(ds2) == 5 and llm.usage.calls == calls and gen.stats["resumed"] == 5


def test_custom_task_from_readme():
    from collections import Counter

    from pydantic import create_model

    import synthgen as sg
    from synthgen.schemas import LabelDef as LD, TaskSchema
    from synthgen.tasks.base import Task, literal_of

    class ToneSchema(TaskSchema):
        tones: list[LD]

    @sg.register_task
    class Tone(Task[ToneSchema]):
        name = "tone"
        schema_cls = ToneSchema

        def answer_model(self, schema):
            return create_model("ToneAnswer", tone=(literal_of([t.name for t in schema.tones]), ...))

        def instruction(self, schema, lang):
            return schema.instruction + "\n" + "\n".join(f"- {t.name}: {t.description}" for t in schema.tones)

        def postprocess(self, text, schema, raw):
            return {"tone": raw.tone}

        def aggregate(self, text, schema, answers):
            c = Counter(a["tone"] for a in answers)
            return {"tone": c.most_common(1)[0][0]}, {"probs": {k: v / len(answers) for k, v in c.items()}}

    class ToneLLM(LLMClient):
        async def _complete(self, messages, **params):
            return json.dumps({"tone": "formal"})

    schema = ToneSchema(name="tone", instruction="Тон текста", tones=[LD(name="formal", description=""), LD(name="casual", description="")])
    ds = Generator(ToneLLM(LLMConfig()), "tone", schema, n_samples=2).run_sync([Document(id="d", text="Уважаемые коллеги")], progress=False)
    assert ds[0].answer == {"tone": "formal"}
    assert export.to_structured(ds)[0]["output_schema"]["properties"]["tone"]["enum"] == ["formal", "casual"]


def test_separate_embedding_server(monkeypatch):
    monkeypatch.setenv("SYNTHGEN_BASE_URL", "http://llm:1/v1")
    monkeypatch.setenv("SYNTHGEN_API_KEY", "llm-key")
    monkeypatch.setenv("SYNTHGEN_EMBED_BASE_URL", "http://emb:2/v1")
    monkeypatch.setenv("SYNTHGEN_EMBED_API_KEY", "emb-key")
    cfg = LLMConfig.from_env(dotenv_path="nonexistent.env")
    assert (cfg.embed_base_url, cfg.embed_api_key) == ("http://emb:2/v1", "emb-key")

    async def clients(config):
        llm = LLMClient(config)
        llm._ensure_loop_bound()
        return llm

    llm = asyncio.run(clients(cfg))
    assert str(llm._client.base_url).startswith("http://llm:1") and llm._client.api_key == "llm-key"
    assert str(llm._embed_client.base_url).startswith("http://emb:2") and llm._embed_client.api_key == "emb-key"
    assert llm._embed_semaphore is not llm._semaphore

    # Без отдельного сервера эмбеддинги идут через основной клиент
    same = asyncio.run(clients(LLMConfig(base_url="http://llm:1/v1")))
    assert same._embed_client is same._client and same._embed_semaphore is same._semaphore
