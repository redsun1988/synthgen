# synthgen

[Русский](README.md) | **English**

`synthgen` is a Python framework that turns a corpus of raw text into synthetic training data using an LLM.
It produces validated examples for seven task types and exports them in formats that are ready for training:

- compact structured-output models (in the spirit of [Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev));
- LLMs fine-tuned with LoRA on examples that include reasoning (`<think>…</think>`);
- classic NLP models (BERT classifiers, token classification, QA) and tabular models (CatBoost, sklearn).

It works with any OpenAI-compatible API: Ollama, vLLM, LM Studio, OpenAI.

---

## Features

| Task | `task` | Answer | Confidence with `n_samples > 1` |
|---|---|---|---|
| Text classification (single/multi-label) | `text_classification` | `{"label": "complaint"}` | class distribution |
| Text regression | `text_regression` | `{"value": 7.5}` | mean, std, all samples |
| Named entity recognition | `ner` | `start/end/text/type` spans (+ normalized `value`) | vote share per entity |
| Entities and relations | `relation_extraction` | entities + `{head, type, tail}` | vote share per relation |
| Extractive QA | `extractive_qa` | questions, including unanswerable ones; single- or multi-span answers | agreement of an independent "reader" |
| Extractive summarization | `extractive_summarization` | verbatim fragments, optionally query-focused | sample agreement (medoid) |
| Tabular data | `tabular` | one table row per document, following a column schema | per-column agreement |

**What every task shares:**

- **Hand-written or LLM-invented schemas.** `schema="auto"` invents a separate task for each document, and `propose_schemas` invents a set of tasks from a corpus sample.
- **Two modes.** `label` annotates the corpus texts. `generate` writes new texts modeled on the corpus, balances the classes, and drops any example whose independent re-labeling misses the target class.
- **Strict validation with self-correction.** The answer is constrained by a JSON Schema. A compact version of the schema, with field descriptions, is also appended to the prompt: Ollama and llama.cpp use the schema only as a grammar, so without it the model never sees what the fields mean. When a programmatic check fails (the span exists in the text, relation argument types, date format, and so on), the error text goes back to the model so it can fix the answer. If that doesn't work, the invalid parts are dropped.
- **No hallucinated offsets.** The LLM returns a text fragment and the framework computes the character offsets. Matching is tolerant to case, quotes, whitespace and Russian word forms ("Набережные Челны" matches "Набережных Челнах").
- **Soft labels for calibration.** `n_samples` sets the number of independent annotations, which yield distributions and confidence scores.
- **Quality control.** An LLM judge with a score threshold, deduplication (MinHash and embedding cosine similarity), class balancing, and a leak-free train/val/test split by source document.
- **Reasoning for SFT.** The reasoning is written after the answer is fixed and validated, so reasoning and answer always agree.
- **Scale.** Async execution with a concurrency limit. Every record is appended to a JSONL checkpoint as it completes, so an interrupted run resumes without repeating requests.

---

## Installation

Requires Python ≥ 3.11.

```bash
git clone <repo> synthgen && cd synthgen
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -e ".[all]"
```

| Extra | Adds |
|---|---|
| (none) | core: `openai`, `pydantic`, `numpy`, `pandas`, `pyarrow`, `python-dotenv`, `tqdm` |
| `hf` | export to `datasets.Dataset` |
| `notebook` | `ipykernel`, `ipywidgets` for the examples |
| `dev` | `pytest`, `build` |
| `all` | all of the above |

To run the example notebook, register a kernel and select it in Jupyter:

```bash
python -m ipykernel install --user --name synthgen --display-name "Python (synthgen)"
```

## Configuration

Copy `.env.example` to `.env`:

```dotenv
SYNTHGEN_BASE_URL=http://localhost:11434/v1    # OpenAI-compatible endpoint
SYNTHGEN_API_KEY=ollama
SYNTHGEN_MODEL=qwen3:30b
SYNTHGEN_EMBED_MODEL=paraphrase-multilingual   # used for deduplication and diverse sampling
SYNTHGEN_EMBED_BASE_URL=                       # optional: a separate embeddings server
SYNTHGEN_EMBED_API_KEY=                        # optional: its key
SYNTHGEN_MAX_CONCURRENCY=4
SYNTHGEN_REASONING_EFFORT=none                 # disables built-in thinking in qwen3/deepseek-r1
```

You can also set the same options in code: `sg.LLMConfig(base_url=..., model=..., max_concurrency=8)`.

The key is sent as `Authorization: Bearer <key>`. Embeddings can come from a different server.
If `embed_base_url` / `embed_api_key` are not set, the main `base_url` / `api_key` are used:

```python
sg.LLMConfig(
    base_url="https://llm.example.com/v1", api_key="sk-llm", model="qwen3:30b",
    embed_base_url="http://embed.example.com:8080/v1", embed_api_key="sk-emb", embed_model="bge-m3",
)
```

A separate embeddings server gets its own `max_concurrency` limit, so embedding calls do not take chat-model slots.

> For thinking models (qwen3, deepseek-r1), keep `SYNTHGEN_REASONING_EFFORT=none`.
> With built-in reasoning turned on, a single call takes minutes instead of seconds. synthgen generates SFT reasoning as a separate step.

---

## Quick start

```python
import synthgen as sg

llm = sg.LLMClient(sg.LLMConfig.from_env())
docs = list(sg.chunk_documents(sg.read_jsonl("data/sample_corpus.jsonl")))

schema = sg.ClassificationSchema(
    name="bank_review_intent",
    instruction="Determine the main purpose of the bank customer's message.",
    labels=[
        sg.LabelDef(name="complaint", description="complaint about a problem"),
        sg.LabelDef(name="praise", description="gratitude, positive feedback"),
        sg.LabelDef(name="question", description="question or request for advice"),
    ],
)

gen = sg.Generator(llm, "text_classification", schema, n_samples=3, output_path="output/intent.jsonl")
ds = await gen.run(docs)          # in Jupyter; in a plain script: ds = gen.run_sync(docs)

print(ds[0].answer, ds[0].confidence)
# {'label': 'complaint'} {'label_probs': {'complaint': 1.0, 'praise': 0.0, 'question': 0.0}}
```

[`notebooks/examples.ipynb`](notebooks/examples.ipynb) has complete examples for every task. The notebook is written in Russian.

---

## Core concepts

```
Document ──► task schema (hand-written | propose_schemas | schema="auto")
         ──► text (mode="label": corpus text | mode="generate": new text modeled on the corpus)
         ──► LLM annotation under a JSON Schema ─► validation ─► self-correction ─► repair
         ──► n_samples annotations ─► aggregation ─► soft labels / confidence
         ──► [LLM judge] ─► [reasoning] ─► Record ─► JSONL checkpoint
         ──► export: structured (Jev) · chat/SFT · BIO/CoNLL · SQuAD v2 · Parquet · HF datasets
```

| Object | Purpose |
|---|---|
| `Document` | corpus unit: `id`, `text`, `meta` |
| `*Schema` | task definition: classes, entity and relation types, table columns, and so on |
| `Task` | how to annotate under a schema, validate the answer and aggregate samples |
| `Generator` | the pipeline: planning, concurrent execution, balancing, judge, reasoning, checkpoints |
| `Record` | canonical record: text, schema, answer, confidence, reasoning, quality, metadata |
| `Dataset` | a set of records: persistence, statistics, leak-free `split` |

### `Generator` parameters

```python
sg.Generator(
    llm, task, schema,              # task: name or Task instance; schema: a schema | a list of schemas | "auto"
    mode="label",                   # "label" | "generate"
    n_samples=1,                    # >1 gives soft labels and confidence
    temperature=0.7,
    per_document=1,                 # generate: new texts per seed document
    rebalance_rounds=2,             # generate: extra rounds that top up under-represented classes
    judge=False, min_judge_score=3, # LLM judge
    reasoning=False,                # reasoning for SFT
    output_path=None,               # resumable JSONL checkpoint
)
```

After a run, `gen.stats` shows how many examples were accepted, how many were rejected and why, and how many failed.
Token usage and retry counts are in `llm.usage`.

---

## Task examples

<details>
<summary><b>NER with normalized values</b></summary>

```python
schema = sg.NERSchema(
    name="news_ner", instruction="Extract named entities.",
    entity_types=[sg.LabelDef(name="PER", description="people"),
                  sg.LabelDef(name="ORG", description="organizations"),
                  sg.LabelDef(name="DATE", description="dates"),
                  sg.LabelDef(name="MONEY", description="amounts of money")],
    with_attributes=True,       # value: ISO date, numeric amount, canonical name form
    allow_nested=False,
)
ds = await sg.Generator(llm, "ner", schema, n_samples=3).run(docs)
# {"start": 44, "end": 56, "text": "Colette Kress", "type": "PER", "value": "Colette Kress", "confidence": 1.0}
sg.export.to_conll(ds, "output/ner.conll")
```
</details>

<details>
<summary><b>Relations between entities</b></summary>

```python
schema = sg.RelationSchema(
    name="business", instruction="Extract the parties of business events and the links between them.",
    entity_types=[sg.LabelDef(name="PER", description="people"), sg.LabelDef(name="ORG", description="organizations")],
    relation_types=[sg.RelationDef(name="works_for", description="the person holds a position at the organization",
                                   head_types=["PER"], tail_types=["ORG"])],
)
ds = await sg.Generator(llm, "relation_extraction", schema, n_samples=3).run(docs)
# relations: [{"head": 2, "tail": 0, "type": "works_for", "head_text": "Christian Bruch", "tail_text": "Siemens Energy", "confidence": 1.0}]
```
</details>

<details>
<summary><b>Extractive QA with unanswerable questions</b></summary>

```python
schema = sg.QASchema(name="facts", instruction="Questions about facts.", n_questions=4,
                     unanswerable_ratio=0.25, allow_multi_span=True, question_focus="dates and amounts")
ds = await sg.Generator(llm, "extractive_qa", schema, n_samples=3).run(docs)
squad = sg.export.to_squad(ds)
```
With `n_samples > 1`, an independent "reader" checks every question. Questions where the reader disagrees with
the generator are dropped. For example, the generator may mark a question as unanswerable even though the text contains the answer.
</details>

<details>
<summary><b>Query-focused extractive summarization</b></summary>

```python
schema = sg.SummarizationSchema(name="risks", instruction="Summary for a risk analyst.",
                                query="risks and negative consequences", max_spans=3)
ds = await sg.Generator(llm, "extractive_summarization", schema, n_samples=3).run(docs)
```
</details>

<details>
<summary><b>Tabular data from documents</b></summary>

```python
schema = sg.TabularSchema(
    name="flats", instruction="Extract apartment attributes from the listing.", target="price_rub",
    columns=[sg.ColumnDef(name="rooms", dtype="int", description="number of rooms, studio = 0"),
             sg.ColumnDef(name="area_m2", dtype="float", description="floor area", unit="m²"),
             sg.ColumnDef(name="wall", dtype="category", description="wall material",
                          categories=["brick", "panel", "monolith", "other"]),
             sg.ColumnDef(name="price_rub", dtype="float", description="price, RUB")],
)
real = await sg.Generator(llm, "tabular", schema, n_samples=3).run(listings)
synthetic = await sg.Generator(llm, "tabular", schema, mode="generate", per_document=5).run(listings)
df = sg.export.to_table(real + synthetic)      # dtypes: Int64, float, category, datetime
df.to_parquet("output/flats.parquet")
```
Values missing from the text stay `null`; the model does not make them up.
</details>

<details>
<summary><b>Many different schemas for a Jev-style model</b></summary>

```python
# A separate task for each document
ds = await sg.Generator(llm, "text_classification", "auto", n_samples=5).run(docs)

# Or a set of tasks invented from a corpus sample, assigned to documents in turn
task = sg.get_task("text_regression")
schemas = await task.propose_schemas(llm, [d.text for d in docs[:20]], k=5, lang="en")
ds = await sg.Generator(llm, task, schemas, n_samples=5).run(docs)

sg.export.write_jsonl(sg.export.to_structured(ds), "output/structured.jsonl")
# {"instruction": ..., "input": ..., "output_schema": {JSON Schema}, "output": {...}, "probabilities": {...}}
```
Each schema needs texts it fits. For example, a "customer satisfaction" schema should not be applied to news articles.
Invent schemas from the same texts that will later be annotated with them.
</details>

---

## Export

| Function | Format | Use |
|---|---|---|
| `Dataset.to_jsonl` | canonical `Record`s | storage, reprocessing |
| `export.to_structured` | input, output JSON Schema, output, probabilities | Jev-style models |
| `export.to_chat` | `messages` with `<think>…</think>` | SFT/LoRA with TRL, Unsloth, axolotl |
| `export.to_conll` / `to_bio` | BIO tags | token classification |
| `export.to_squad` | SQuAD v2.0 | extractive QA |
| `export.to_table` | typed `pandas.DataFrame` | CatBoost, sklearn, Parquet/CSV |
| `export.to_hf` | `datasets.Dataset` | HuggingFace (`pip install synthgen[hf]`) |

```python
parts = ds.split(val=0.1, test=0.1)   # by source document: chunks of one document never end up in different splits
unique, stats = await sg.deduplicate(list(ds), llm=llm, minhash_threshold=0.7, embedding_threshold=0.95)
```

---

## Custom tasks

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

You can also override `validate` (a list of errors for self-correction), `repair`,
`generation_targets` / `generation_hint` / `target_matches` (for `generate` mode) and `proposal_hint` (for auto-invented schemas).

---

## Project layout

```
src/synthgen/
  config.py        LLMConfig, .env loading
  llm.py           async client: JSON Schema, validation, self-correction, repair, concurrency limit, embeddings
  corpus.py        corpus adapters, chunking, language detection, diverse sampling
  spans.py         span matching: case, quotes, whitespace, word forms
  schemas.py       task schemas
  tasks/           classification, ner, extractive (QA, summarization, tabular), base
  pipeline.py      Generator
  quality.py       LLM judge, deduplication, balancing
  reasoning.py     post-hoc reasoning
  export.py        exporters
  dataset.py       Dataset
  viz.py           span highlighting in Jupyter
notebooks/examples.ipynb   examples for every task
data/sample_corpus.jsonl   demo corpus (RU/EN)
tests/                     tests that don't need an LLM
```

## Tests and packaging

```bash
pytest                      # offline tests with a fake LLM client
python -m build             # wheel and sdist in dist/
```

## Troubleshooting

- **`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`** when pandas is imported in Jupyter.
  The kernel runs in an environment where an old `pyarrow`, `numexpr` or `bottleneck` conflicts with NumPy 2 (the base Anaconda environment is a common case).
  Use the dedicated environment from the Installation section and the *Python (synthgen)* kernel.
- **Every call takes minutes.** A thinking model has its built-in reasoning turned on. Check that `SYNTHGEN_REASONING_EFFORT=none`.
- **Many `failed` entries in `gen.stats`.** The model struggles with the schema. Simplify the descriptions, use fewer classes,
  or switch to a larger model. WARNING-level logs show the details.

## Roadmap

Coreference in NER, calibration via logprobs, separate models for the generator and the judge,
table synthesis from a causal graph (DAG).
