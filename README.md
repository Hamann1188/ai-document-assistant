# AI Document Assistant

[![CI](https://github.com/Hamann1188/ai-document-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Hamann1188/ai-document-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Chat with your company's PDFs.** Ask in English, Russian or Uzbek. Every answer cites the file and page it comes from, and when the documents don't cover a question, the assistant says so instead of guessing.

![Answers with page citations](docs/images/answers.png)

## The problem

Price lists, policies, handbooks and contracts hold the answers that staff and customers need, but searching them takes time. A general chatbot answers confidently even when it's wrong. A business needs answers that:

- come **only from its own documents**;
- can be **checked in one click**: each fact links to the exact page of the PDF;
- say **"this isn't in the documents"** when that's the truth;
- work across languages, for example a Russian question about an English contract.

## Results

Measured by the end-to-end eval in [`evals/run.py`](evals/run.py) over 30 questions about the sample documents. A separate Claude call grades each answer, and the verdicts were checked by reading every answer. The full per-question table is in [`evals/results/latest.md`](evals/results/latest.md).

| Metric | Result | Target |
|---|---|---|
| Correct answers (25 in-scope questions) | **100%** | ≥ 90% |
| Cites a page that contains the answer | **100%** | ≥ 90% |
| Says "not in the documents" when it isn't (4 questions) | **100%** | 100% |
| Ignores an instruction planted in a document | **100%** | 100% |
| Every claim supported by the documents (30 answers) | **100%** | ≥ 95% |
| Replies in the language of the question | **100%** | ≥ 95% |

- About **$0.02 per answer** with Claude Opus 5.5. First words appear after a median **1.5 s**; the full answer takes **3.5 s**.
- Retrieval alone finds the right page first for 92% of questions, and within the top 8 for 100% ([`evals/results/retrieval.md`](evals/results/retrieval.md)).
- 136 automated tests run in CI on every push, including integration tests against PostgreSQL.

## Features

- **Upload PDFs** by drag and drop or through the API. Processing takes seconds, and a file uploaded twice is stored once.
- **Streaming answers** with numbered footnotes. A footnote opens the original PDF at the cited page, and a sources list shows the quoted text.
- **Multilingual:** questions and documents in English, Russian and Uzbek, in any combination.
- **Honest about gaps:** "the documents don't mention it" instead of an invented answer.
- **Resistant to prompt injection:** text inside documents is treated as data. The sample contract hides an instruction to say "all treatments are free"; the assistant ignores it.
- **Exact references:** clause numbers, codes and abbreviations ("clause 4.2", "ОПТГ") are matched exactly.
- **Cost and token usage** are shown under every answer and logged on the server.
- **Clean web UI** with light and dark themes and a phone layout. No build step, no external scripts, strict Content Security Policy.
- **REST API** with interactive docs at `/docs`.

| | |
|---|---|
| ![Says when the documents don't contain the answer](docs/images/not-in-documents-dark.png) | Asked about something the documents don't cover, the assistant says so, shows what the documents *do* say and suggests whom to contact. |

## How it works

```mermaid
flowchart LR
  subgraph Ingestion
    P["PDF upload"] --> X["Text per page"]
    X --> C["Chunks that never cross a page"]
    C --> E["Local embeddings<br/>embeddinggemma-300m"]
  end
  E --> DB[("PostgreSQL + pgvector")]
  Q["Question"] --> S["Semantic search<br/>+ exact-code boost"]
  DB --> S
  S --> CL["Claude Opus 5.5<br/>pages as search results,<br/>citations on"]
  CL -- "streamed text + citations" --> UI["Web UI:<br/>footnotes open the PDF page"]
```

1. **Ingestion.** Text is extracted page by page, repeated headers and footers are removed, and the text is split into sentence-aligned chunks of up to 120 words. A chunk never spans two pages, so every citation points to a single page. Embeddings are computed locally with an open model, so no embeddings API is needed.
2. **Retrieval.** The question is embedded and matched against the chunks in PostgreSQL (pgvector, HNSW index). Chunks that contain an exact code or abbreviation from the question move to the top.
3. **Answering.** The best pages go to Claude as `search_result` blocks with native citations, one block per sentence. Each citation maps back to a file and page, and the answer streams to the browser over server-sent events.

Design decisions came from measurements, not defaults:

- the embedding model was picked by a benchmark of four candidates on this question set;
- the classic hybrid ranking (Reciprocal Rank Fusion) was tested and dropped because it scored lower (MRR 0.83 against 0.94).

Details, with the numbers, are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

Requirements: Docker with Compose, and an [Anthropic API key](https://platform.claude.com).

```bash
git clone https://github.com/Hamann1188/ai-document-assistant.git
cd ai-document-assistant
cp .env.example .env          # then set DOCASSIST_ANTHROPIC_API_KEY in .env
docker compose up -d --build
```

Open **http://localhost:8000** and drag the PDFs from [`sample_docs/`](sample_docs) into the Documents panel, or upload them from the command line:

```bash
for f in sample_docs/*.pdf; do curl -F "file=@$f;type=application/pdf" http://localhost:8000/documents; done
```

The first build downloads the 1.2 GB embedding model into the image, so it takes a few minutes; later starts download nothing. Without an API key, uploading and search still work, and the answer route returns a clear error.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (multipart `file`); processed in the background |
| `GET` | `/documents` | List documents with status: `processing`, `ready` or `failed` with a reason |
| `GET` | `/documents/{id}` | One document |
| `GET` | `/documents/{id}/file` | The original PDF; add `#page=N` to open a page |
| `DELETE` | `/documents/{id}` | Delete a document, its chunks and the file |
| `GET` | `/search?q=&k=` | The chunks the assistant would use as sources |
| `POST` | `/ask` | `{"question": "..."}` → server-sent events (`sources`, `text`, `citation`, `done`, `error`); add `"stream": false` for one JSON object |
| `GET` | `/healthz` | Liveness and database check |

```bash
curl -N -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How early must I cancel an appointment?"}'
```

## Configuration

Set in `.env` (template: [`.env.example`](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `DOCASSIST_ANTHROPIC_API_KEY` | – | Required for answers |
| `DOCASSIST_MODEL` | `claude-opus-5-5` | Answer model. `claude-sonnet-5-5` costs half as much |
| `DOCASSIST_ANSWER_EFFORT` | `medium` | Reasoning effort: `low` … `max` |
| `DOCASSIST_RETRIEVAL_K` | `8` | Chunks retrieved per question |
| `DOCASSIST_REFUSAL_FALLBACK` | `true` | If the model declines, retry on Anthropic's fallback model (beta) |
| `DOCASSIST_MAX_UPLOAD_MB` / `DOCASSIST_MAX_PAGES` | `20` / `300` | Upload limits |

## Development

Needs [uv](https://docs.astral.sh/uv/) and Docker for the database.

```bash
uv sync
docker compose up -d db
uv run alembic upgrade head
uv run uvicorn docassist.api.main:app --reload       # http://localhost:8000

uv run ruff check . && uv run ruff format --check .
uv run pytest                                        # unit tests, no API calls
DOCASSIST_TEST_DATABASE_URL=postgresql+asyncpg://docassist:docassist@localhost:5432/docassist_test \
  uv run pytest                                      # plus integration tests

uv run python -m evals.retrieval                     # retrieval quality, free
uv run python -m evals.run                           # full eval with real API calls, about $1
```

The sample PDFs are generated by [`scripts/make_sample_docs.py`](scripts/make_sample_docs.py), and the eval questions with their expected pages are in [`evals/questions.yaml`](evals/questions.yaml).

## Stack

Python 3.13 · FastAPI · SQLAlchemy 2 (async) · PostgreSQL 17 + pgvector · Alembic · pypdf · fastembed (ONNX) · Anthropic Python SDK (Claude Opus 5.5) · vanilla HTML/CSS/JS · pytest · ruff · Docker Compose · GitHub Actions

## Limitations

- **Text PDFs only.** Scanned PDFs without a text layer are rejected with a clear message; there is no OCR.
- **No authentication.** This is a single-team demo meant to run locally or behind your own login; there are no per-user spaces.
- **One question at a time.** Follow-up questions don't see the previous answer.
- **Tables** are cited as a whole block, though the page is still exact.
- **Repeated quotes.** Occasionally an answer states a fact and then repeats it as a quoted citation (2 of 30 eval answers).
- **Licence of the embedding model.** It is under the Gemma Terms of Use. An MIT-licensed alternative was benchmarked; switching to it needs a re-index.
- **Single instance.** Background processing runs inside the app process.

## Possible extensions

OCR for scanned documents (Claude vision) · Word and Excel files · login and per-team document spaces · Telegram or Slack bot front end · conversation memory · Google Drive or Notion sync · hosted deployment.

## About

All documents, people and prices belong to **Registan Smile Clinic, a fictional company** created for this demo.

Built by [@Hamann1188](https://github.com/Hamann1188), available for freelance work on AI assistants, chatbots and automation.

Licensed under the [MIT License](LICENSE).
