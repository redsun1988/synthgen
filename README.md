# synthgen

**Русский** | [English](README.en.md)

`synthgen` — Python-фреймворк для генерации синтетических обучающих данных из корпуса сырых текстов с помощью LLM.
Он превращает неразмеченные тексты в проверенные примеры для семи типов задач и выгружает их в форматы,
которые сразу подходят для обучения:

- компактных моделей со структурированным выходом (в духе [Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev));
- LLM, дообучаемых через LoRA на примерах с рассуждениями (`<think>…</think>`);
- классических NLP-моделей (BERT-классификаторы, token classification, QA) и табличных моделей (CatBoost, sklearn).

Работает с любым OpenAI-совместимым API: Ollama, vLLM, LM Studio, OpenAI.

---

## Возможности

| Задача | `task` | Ответ | Уверенность при `n_samples > 1` |
|---|---|---|---|
| Классификация текстов (single/multi-label) | `text_classification` | `{"label": "complaint"}` | распределение по классам |
| Регрессия по тексту | `text_regression` | `{"value": 7.5}` | среднее, std, все сэмплы |
| Выделение именованных сущностей | `ner` | спаны `start/end/text/type` (+ нормализованное `value`) | доля голосов за каждую сущность |
| Сущности и отношения | `relation_extraction` | сущности + `{head, type, tail}` | доля голосов за отношение |
| Экстрактивный QA | `extractive_qa` | вопросы, в т.ч. без ответа; ответы из одного или нескольких спанов | согласие независимого «читателя» |
| Экстрактивная суммаризация | `extractive_summarization` | дословные фрагменты, в т.ч. по запросу | согласие сэмплов (медоида) |
| Табличные данные | `tabular` | строка таблицы по схеме колонок | согласие по каждой колонке |

**Что общего у всех задач:**

- **Схемы задаются вручную или придумываются LLM.** `schema="auto"` генерирует свою задачу для каждого документа, а `propose_schemas` — набор задач по выборке корпуса.
- **Два режима.** `label` размечает тексты корпуса. `generate` пишет новые тексты по мотивам корпуса, балансирует классы и отбрасывает примеры, у которых повторная разметка не совпала с целевым классом.
- **Строгая валидация с самоисправлением.** Ответ ограничен JSON Schema. Компактная схема с описаниями полей дописывается и в промпт: Ollama и llama.cpp используют схему только как грамматику, и без этого модель не видит, что означают поля. Программные проверки (спан есть в тексте, типы аргументов отношения, формат даты и т.п.) при ошибке возвращают модели текст ошибки, и она исправляет ответ. Если исправить не удалось, невалидные части отбрасываются.
- **Спаны без галлюцинаций смещений.** LLM возвращает фрагмент текста, а символьные смещения вычисляет фреймворк. Поиск учитывает регистр, кавычки, пробелы и русские словоформы («Набережные Челны» → «Набережных Челнах»).
- **Мягкие метки для калибровки.** Параметр `n_samples` задаёт число независимых разметок, из них получаются распределения и уверенности.
- **Контроль качества.** Доступны LLM-судья с порогом оценки, дедупликация (MinHash и косинусная близость эмбеддингов), баланс классов и разбиение на train/val/test по исходному документу, без утечек.
- **Рассуждения для SFT.** Рассуждение пишется после того, как ответ зафиксирован и проверен, поэтому рассуждение и ответ всегда согласованы.
- **Масштаб.** Генерация асинхронная, с ограничением параллельности. Каждая запись сразу дописывается в JSONL-чекпоинт, и прерванный запуск продолжается без повторных запросов.

---

## Установка

Требуется Python ≥ 3.11.

```bash
git clone <repo> synthgen && cd synthgen
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -e ".[all]"
```

| Extra | Что добавляет |
|---|---|
| (без extras) | ядро: `openai`, `pydantic`, `numpy`, `pandas`, `pyarrow`, `python-dotenv`, `tqdm` |
| `hf` | экспорт в `datasets.Dataset` |
| `notebook` | `ipykernel`, `ipywidgets` для примеров |
| `dev` | `pytest`, `build` |
| `all` | всё перечисленное |

Для ноутбука с примерами зарегистрируйте ядро и выберите его в Jupyter:

```bash
python -m ipykernel install --user --name synthgen --display-name "Python (synthgen)"
```

## Настройка

Скопируйте `.env.example` в `.env`:

```dotenv
SYNTHGEN_BASE_URL=http://192.168.1.12:11434/v1   # OpenAI-совместимый эндпоинт
SYNTHGEN_API_KEY=ollama
SYNTHGEN_MODEL=qwen3:30b
SYNTHGEN_EMBED_MODEL=paraphrase-multilingual      # для дедупликации и отбора разнообразных текстов
SYNTHGEN_EMBED_BASE_URL=                          # необязательно: отдельный сервер эмбеддингов
SYNTHGEN_EMBED_API_KEY=                           # необязательно: ключ для него
SYNTHGEN_MAX_CONCURRENCY=4
SYNTHGEN_REASONING_EFFORT=none                    # отключает встроенный thinking у qwen3/deepseek-r1
```

Эти же параметры можно передать в коде: `sg.LLMConfig(base_url=..., model=..., max_concurrency=8)`.

Ключ передаётся в заголовке `Authorization: Bearer <ключ>`. Эмбеддинги можно брать с другого сервера.
Если `embed_base_url` / `embed_api_key` не заданы, используются основные `base_url` / `api_key`:

```python
sg.LLMConfig(
    base_url="https://llm.example.com/v1", api_key="sk-llm", model="qwen3:30b",
    embed_base_url="http://embed.example.com:8080/v1", embed_api_key="sk-emb", embed_model="bge-m3",
)
```

У эмбеддингов на отдельном сервере свой лимит `max_concurrency`, поэтому они не занимают слоты чат-модели.

> Для thinking-моделей (qwen3, deepseek-r1) оставьте `SYNTHGEN_REASONING_EFFORT=none`.
> Со встроенным рассуждением один вызов занимает минуты вместо секунд. Рассуждения для SFT фреймворк генерирует отдельным шагом.

---

## Быстрый старт

```python
import synthgen as sg

llm = sg.LLMClient(sg.LLMConfig.from_env())
docs = list(sg.chunk_documents(sg.read_jsonl("data/sample_corpus.jsonl")))

schema = sg.ClassificationSchema(
    name="bank_review_intent",
    instruction="Определи основную цель обращения клиента банка.",
    labels=[
        sg.LabelDef(name="complaint", description="жалоба на проблему"),
        sg.LabelDef(name="praise", description="благодарность, положительный отзыв"),
        sg.LabelDef(name="question", description="вопрос или просьба о консультации"),
    ],
)

gen = sg.Generator(llm, "text_classification", schema, n_samples=3, output_path="output/intent.jsonl")
ds = await gen.run(docs)          # в Jupyter; в обычном скрипте: ds = gen.run_sync(docs)

print(ds[0].answer, ds[0].confidence)
# {'label': 'complaint'} {'label_probs': {'complaint': 1.0, 'praise': 0.0, 'question': 0.0}}
```

Полный набор примеров для всех задач находится в [`notebooks/examples.ipynb`](notebooks/examples.ipynb).

---

## Основные понятия

```
Document ──► схема задачи (вручную | propose_schemas | schema="auto")
         ──► текст (mode="label": текст корпуса | mode="generate": новый текст по мотивам)
         ──► разметка LLM по JSON Schema ─► валидация ─► самоисправление ─► repair
         ──► n_samples разметок ─► агрегация ─► мягкие метки / уверенность
         ──► [LLM-судья] ─► [рассуждение] ─► Record ─► JSONL-чекпоинт
         ──► экспорт: structured (Jev) · chat/SFT · BIO/CoNLL · SQuAD v2 · Parquet · HF datasets
```

| Объект | Назначение |
|---|---|
| `Document` | единица корпуса: `id`, `text`, `meta` |
| `*Schema` | описание задачи: классы, типы сущностей и отношений, колонки таблицы и т.п. |
| `Task` | как размечать по схеме, как валидировать ответ и сводить сэмплы |
| `Generator` | пайплайн: планирование, параллельное выполнение, баланс, судья, рассуждения, чекпоинты |
| `Record` | каноническая запись: текст, схема, ответ, уверенность, рассуждение, качество, метаданные |
| `Dataset` | набор записей: сохранение, статистика, `split` без утечек |

### Параметры `Generator`

```python
sg.Generator(
    llm, task, schema,              # task: имя или экземпляр Task; schema: схема | список схем | "auto"
    mode="label",                   # "label" | "generate"
    n_samples=1,                    # >1 — мягкие метки и уверенность
    temperature=0.7,
    per_document=1,                 # generate: сколько новых текстов на документ-образец
    rebalance_rounds=2,             # generate: догенерация недобравших классов
    judge=False, min_judge_score=3, # LLM-судья
    reasoning=False,                # рассуждение для SFT
    output_path=None,               # JSONL-чекпоинт с возобновлением
)
```

После запуска `gen.stats` показывает, сколько примеров принято, отброшено по разным причинам и сколько не удалось сгенерировать.
Расход токенов и число повторных попыток лежат в `llm.usage`.

---

## Примеры по задачам

<details>
<summary><b>NER с нормализацией значений</b></summary>

```python
schema = sg.NERSchema(
    name="news_ner", instruction="Выдели именованные сущности.",
    entity_types=[sg.LabelDef(name="PER", description="люди"),
                  sg.LabelDef(name="ORG", description="организации"),
                  sg.LabelDef(name="DATE", description="даты"),
                  sg.LabelDef(name="MONEY", description="денежные суммы")],
    with_attributes=True,       # value: дата в ISO, сумма числом, имя в именительном падеже
    allow_nested=False,
)
ds = await sg.Generator(llm, "ner", schema, n_samples=3).run(docs)
# {"start": 159, "end": 175, "text": "Дмитрия Полищука", "type": "PER", "value": "Дмитрий Полищук", "confidence": 1.0}
sg.export.to_conll(ds, "output/ner.conll")
```
</details>

<details>
<summary><b>Отношения между сущностями</b></summary>

```python
schema = sg.RelationSchema(
    name="business", instruction="Выдели участников деловых событий и связи между ними.",
    entity_types=[sg.LabelDef(name="PER", description="люди"), sg.LabelDef(name="ORG", description="организации")],
    relation_types=[sg.RelationDef(name="works_for", description="человек работает в организации",
                                   head_types=["PER"], tail_types=["ORG"])],
)
ds = await sg.Generator(llm, "relation_extraction", schema, n_samples=3).run(docs)
# relations: [{"head": 3, "tail": 0, "type": "works_for", "head_text": "Эльвира Набиуллина", "tail_text": "Центральный банк России", "confidence": 1.0}]
```
</details>

<details>
<summary><b>Экстрактивный QA с вопросами без ответа</b></summary>

```python
schema = sg.QASchema(name="facts", instruction="Вопросы по фактам.", n_questions=4,
                     unanswerable_ratio=0.25, allow_multi_span=True, question_focus="сроки и суммы")
ds = await sg.Generator(llm, "extractive_qa", schema, n_samples=3).run(docs)
squad = sg.export.to_squad(ds)
```
При `n_samples > 1` каждый вопрос проверяется независимым «читателем». Вопросы, где читатель не согласен
с генератором, отбрасываются. Пример такой ошибки генератора: вопрос помечен «без ответа», а ответ в тексте есть.
</details>

<details>
<summary><b>Экстрактивная суммаризация по запросу</b></summary>

```python
schema = sg.SummarizationSchema(name="risks", instruction="Выжимка для риск-аналитика.",
                                query="риски и негативные последствия", max_spans=3)
ds = await sg.Generator(llm, "extractive_summarization", schema, n_samples=3).run(docs)
```
</details>

<details>
<summary><b>Табличные данные из документов</b></summary>

```python
schema = sg.TabularSchema(
    name="flats", instruction="Извлеки параметры квартиры из объявления.", target="price_rub",
    columns=[sg.ColumnDef(name="rooms", dtype="int", description="число комнат, студия = 0"),
             sg.ColumnDef(name="area_m2", dtype="float", description="площадь", unit="м²"),
             sg.ColumnDef(name="wall", dtype="category", description="материал стен",
                          categories=["brick", "panel", "monolith", "other"]),
             sg.ColumnDef(name="price_rub", dtype="float", description="цена, руб.")],
)
real = await sg.Generator(llm, "tabular", schema, n_samples=3).run(listings)
synthetic = await sg.Generator(llm, "tabular", schema, mode="generate", per_document=5).run(listings)
df = sg.export.to_table(real + synthetic)      # типы: Int64, float, category, datetime
df.to_parquet("output/flats.parquet")
```
Значения, которых нет в тексте, остаются `null` и не выдумываются.
</details>

<details>
<summary><b>Много разных схем для модели в духе Jev</b></summary>

```python
# Своя задача для каждого документа
ds = await sg.Generator(llm, "text_classification", "auto", n_samples=5).run(docs)

# Или набор задач по выборке корпуса, которые затем раздаются документам по кругу
task = sg.get_task("text_regression")
schemas = await task.propose_schemas(llm, [d.text for d in docs[:20]], k=5, lang="ru")
ds = await sg.Generator(llm, task, schemas, n_samples=5).run(docs)

sg.export.write_jsonl(sg.export.to_structured(ds), "output/structured.jsonl")
# {"instruction": ..., "input": ..., "output_schema": {JSON Schema}, "output": {...}, "probabilities": {...}}
```
Каждой схеме нужны подходящие тексты. Например, схему «удовлетворённость клиента» не стоит применять к новостям.
Придумывайте схемы по тем же текстам, которые потом будут по ним размечаться.
</details>

---

## Экспорт

| Функция | Формат | Для чего |
|---|---|---|
| `Dataset.to_jsonl` | канонические `Record` | хранение, повторная обработка |
| `export.to_structured` | вход, JSON Schema выхода, выход, вероятности | модели в духе Jev |
| `export.to_chat` | `messages` с `<think>…</think>` | SFT/LoRA через TRL, Unsloth, axolotl |
| `export.to_conll` / `to_bio` | BIO-теги | token classification |
| `export.to_squad` | SQuAD v2.0 | экстрактивный QA |
| `export.to_table` | `pandas.DataFrame` с типами | CatBoost, sklearn, Parquet/CSV |
| `export.to_hf` | `datasets.Dataset` | HuggingFace (`pip install synthgen[hf]`) |

```python
parts = ds.split(val=0.1, test=0.1)   # по исходному документу: чанки одного документа не попадут в разные части
unique, stats = await sg.deduplicate(list(ds), llm=llm, minhash_threshold=0.7, embedding_threshold=0.95)
```

---

## Своя задача

```python
from pydantic import BaseModel, create_model
from synthgen.tasks.base import Task, literal_of
from synthgen.schemas import TaskSchema, LabelDef

class ToneSchema(TaskSchema):
    tones: list[LabelDef]

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
        from collections import Counter
        c = Counter(a["tone"] for a in answers)
        return {"tone": c.most_common(1)[0][0]}, {"probs": {k: v / len(answers) for k, v in c.items()}}
```

Можно также переопределить `validate` (список ошибок для самоисправления), `repair`,
`generation_targets` / `generation_hint` / `target_matches` (для режима `generate`) и `proposal_hint` (для автоматических схем).

---

## Структура проекта

```
src/synthgen/
  config.py        LLMConfig, чтение .env
  llm.py           async-клиент: JSON Schema, валидация, самоисправление, repair, лимит параллельности, эмбеддинги
  corpus.py        адаптеры корпуса, нарезка на чанки, определение языка, отбор разнообразных документов
  spans.py         поиск спанов: регистр, кавычки, пробелы, словоформы
  schemas.py       схемы задач
  tasks/           classification, ner, extractive (QA, суммаризация, таблицы), base
  pipeline.py      Generator
  quality.py       LLM-судья, дедупликация, баланс
  reasoning.py     рассуждения постфактум
  export.py        экспортёры
  dataset.py       Dataset
  viz.py           подсветка спанов в Jupyter
notebooks/examples.ipynb   примеры для всех задач
data/sample_corpus.jsonl   демо-корпус (RU/EN)
tests/                     тесты без LLM
```

## Тесты и сборка

```bash
pytest                      # офлайн-тесты на фейковом LLM-клиенте
python -m build             # wheel и sdist в dist/
```

## Решение проблем

- **`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`** при импорте pandas в Jupyter.
  Ядро запущено из окружения, где старые `pyarrow`, `numexpr` или `bottleneck` конфликтуют с NumPy 2 (часто это базовая Anaconda).
  Используйте отдельное окружение из раздела «Установка» и ядро *Python (synthgen)*.
- **Каждый вызов занимает минуты.** У thinking-модели включено встроенное рассуждение. Проверьте `SYNTHGEN_REASONING_EFFORT=none`.
- **Много `failed` в `gen.stats`.** Модель не справляется со схемой. Упростите описания, уменьшите число классов
  или возьмите модель побольше. Подробности видны в логах уровня WARNING.

## Планы

Кореференция в NER, калибровка через logprobs, отдельные модели для генератора и судьи,
синтез таблиц через причинный граф (DAG).
