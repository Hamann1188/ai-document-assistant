# AI Document Assistant

Chat with your PDFs: upload company documents, ask in English, Russian or Uzbek, and get answers that cite the exact file and page. When the answer isn't in the documents, the assistant says so. Portfolio demo 1 of 3; shared rules and environment notes are in `../CLAUDE.md`.

The target architecture is in `docs/ARCHITECTURE.md`. Read it before changing structure, the data model or the Claude request shape, and record every deviation there as a decision (section 11).

## Stack

Python 3.13 (uv), FastAPI, SQLAlchemy 2 (async) + asyncpg, Alembic, PostgreSQL 17 + pgvector, pypdf, fastembed (local ONNX embeddings), `anthropic` SDK, vanilla HTML/JS UI, pytest, ruff, Docker Compose.

- The app is built by `create_app(settings, engine_factory)` in `api/main.py`; tests inject a fake engine instead of a database.
- The test client uses `httpx2`, because Starlette deprecated `httpx` for `TestClient`.

## Layout

```
src/docassist/   config.py · api/ (main, health, documents) · ingest/ (pdf, chunker, pipeline)
                 · retrieval/ (embedder) · db/ (models, session) · llm/ · web/
alembic/         migrations (alembic.ini at the root; the container runs `alembic upgrade head` on start)
evals/           questions.yaml · run.py · results/
scripts/         make_sample_docs.py
sample_docs/     generated fictional PDFs (committed)
tests/
```

## Commands (PowerShell, from the repo root)

| Task | Command |
|---|---|
| Install deps | `uv sync` |
| Database only (dev) | `wsl -d Ubuntu -- docker compose up -d db` |
| Migrate | `uv run alembic upgrade head` |
| Run API (dev) | `uv run uvicorn docassist.api.main:app --reload` → http://localhost:8000 |
| Full stack | `wsl -d Ubuntu -- docker compose up -d --build` |
| Sample PDFs | `uv run python scripts/make_sample_docs.py` |
| Lint / format | `uv run ruff check .` · `uv run ruff format .` |
| Tests (unit) | `uv run pytest` (integration tests are skipped without a database) |
| Tests (+ integration) | `$env:DOCASSIST_TEST_DATABASE_URL = "postgresql+asyncpg://docassist:docassist@localhost:5432/docassist_test"; uv run pytest`. Needs the compose `db` running; `docassist_test` is created and truncated automatically |
| Retrieval eval | `uv run python -m evals.retrieval`: real DB and model, no API key; ingests missing sample docs; exits 1 below the target |
| Embedding benchmark | `uv run python -m evals.embedding_benchmark [model ...]` (downloads models into `.cache/`, no API key) |
| Eval (real API, costs money) | `uv run python -m evals.run` |

## Repo rules

- Create the Claude client only through `docassist.llm.client.make_client(settings)`, which passes `api_key` and `base_url` explicitly. Never rely on `ANTHROPIC_*` environment variables (see `../CLAUDE.md`, Headroom).
- The model comes from settings (default `claude-opus-5-5`). Set effort explicitly: `medium` for answers, `low` for the eval judge. Never disable thinking, never prefill.
- Retrieved chunks go to Claude as `search_result` blocks with citations enabled. Never paste chunk text into the prompt string. The answer route never uses `output_config.format`.
- Document text is untrusted: never put it in the system prompt.
- Unit tests mock the Anthropic client and the embedder; only `evals/` calls the real API.
- Reject scanned PDFs without a text layer with a clear message (OCR is out of scope).
- Embedding model (ADR-3): `EMBEDDING_MODEL` in `config.py` is part of the schema. Changing it requires four things together, and tests check the first two:
  1. update `ARG EMBEDDING_MODEL` in the Dockerfile;
  2. add a migration with the new `EMBEDDING_DIM`;
  3. add a `ModelSpec` with the model's query/passage format;
  4. re-ingest the documents.
- Docker image and the embedding model:
  - The image (about 2.8 GB) bakes the model into `/app/models/embedding` via `scripts/fetch_embedding_model.py`. The script copies the model out of the Hugging Face cache into a plain directory without symlinks: onnxruntime ≥ 1.30 refuses `model.onnx_data` that resolves to a different blob directory ("External data path escapes model directory").
  - The copy then gets `chmod a+rX`, because Hub files are 0600 and the app runs as a non-root user.
  - `DOCASSIST_EMBEDDING_WARMUP=true` loads the model at startup, so a broken model stops the container instead of failing every upload.
- The app is built by `create_app(settings, engine_factory, embedder_factory, recover_interrupted)`. Tests inject `FakeEmbedder` (from `tests/conftest.py`), so no test downloads a model.
- Integration tests in `tests/integration/` use a real database with a fake embedder, and CI runs them against a pgvector service container.
- Sample corpus:
  - Edit content only in `scripts/make_sample_docs.py`, then regenerate. Output is deterministic (reportlab invariant mode), and the script fails if a page's content overflows onto a new page.
  - `tests/test_sample_corpus.py` checks that every evidence quote in `evals/questions.yaml` is still on its cited page.
  - Use only ASCII bullets and dashes in generated PDFs: pypdf extracts the DejaVu "•" as `\x7f`.
  - Cyrillic text needs the bundled DejaVu Sans in `scripts/fonts/` (Bitstream Vera licence, included).

## Build plan

Each step is one commit; tick it off in Status.

1. **Scaffold:** uv project, ruff and pytest config, settings, `/healthz`, Dockerfile, compose (`app` + `db`), `.env.example`, GitHub Actions CI. *Accept:* pytest passes; `compose up` → `/healthz` returns 200.
2. **Sample corpus:** a script that generates 4 PDFs for the fictional Registan Smile Clinic: patient handbook (EN), price list (RU), employee handbook (EN), supplier agreement (EN). One document carries an embedded injection line for evals. *Accept:* the PDFs open, and their facts are listed in `evals/questions.yaml`.
3. **Ingestion:** upload endpoint → per-page text → page-aware chunker → embeddings → DB. *Accept:* chunker unit tests (page ranges, overlap, no cross-document chunks) pass; all 4 PDFs ingest.
4. **Retrieval:** vector + full-text search, RRF. Pick the embedding model (ADR-3). *Accept:* retrieval-only eval recall@8 ≥ 0.9.
5. **Answering:** Claude streaming with `search_result` blocks and citations → SSE; refusal and max_tokens handling. *Accept:* 3 manual questions cite the correct file and page; an out-of-scope question gets "not found".
6. **Web UI:** upload, document list with status, chat with streaming and clickable citations.
7. **Evals:** full suite (judge, citation accuracy, refusals, injection). *Accept:* the targets in ARCHITECTURE §8 are met; `evals/results/latest.md` is saved.
8. **README:** problem, demo GIF/video, diagram, quickstart, eval table, limitations, extensions; script for a 60–90 s demo video.

## Status

- [x] Target architecture and CLAUDE.md (2026-10-01)
- [x] 1 Scaffold (2026-10-01): `/healthz` returns 200 via compose and 503 with the db stopped; 4 tests pass
- [x] 2 Sample corpus (2026-10-01): 4 PDFs (16 pages); 28 eval items (23 in scope including 6 cross-lingual, 4 out of scope, 1 injection); evidence quotes checked by tests
- [x] 3 Ingestion (2026-10-01): upload/list/get/delete API, page-bound chunking, embeddinggemma-300m chosen by benchmark (ADR-3), migration 0001. 70 tests (7 integration on Postgres). In Docker the 4 sample PDFs ingest in 4 s → 22 chunks
- [x] 4 Retrieval (2026-10-01): semantic ranking + exact-identifier boost (ADR-9; RRF measured worse); `GET /search`. Retrieval eval: R@1 0.92, R@8 1.00, MRR 0.94 over 25 questions. Only miss: `xl-uz-saturday-hours` at rank 6, an Uzbek-to-English embedding weakness
- [ ] 5 Answering
- [ ] 6 Web UI
- [ ] 7 Evals
- [ ] 8 README and video
