# Architecture — AI Document Assistant

## 1. Problem and scope

Teams keep answers in PDFs (policies, price lists, contracts), and staff lose time searching them. The assistant answers questions from those documents only. It cites the file and page for every claim, and says plainly when the documents don't contain the answer.

**Goals**

- Upload PDFs through a web UI or the API. Ingestion is idempotent: the same file is never stored twice.
- Questions and answers in English, Russian and Uzbek; sources may be in any of them.
- Every factual statement is backed by a citation (file, page, quoted text).
- Quality is measured by an eval suite whose results are published in the README.
- One-command start: `docker compose up`.

**Non-goals (v1)** — listed in the README as paid extensions: OCR for scanned PDFs, DOCX/XLSX, authentication and multi-tenancy, conversation memory across questions, cloud hosting.

## 2. System overview

```mermaid
flowchart LR
  U[Browser UI] -- upload PDF --> API[FastAPI]
  U -- question --> API
  API -- background task --> ING[Ingestion: pypdf → page-aware chunker → embedder]
  ING --> DB[(PostgreSQL 17 + pgvector)]
  API -- hybrid search --> DB
  API -- search_result blocks + citations --> C[Claude API]
  C -- streamed text + citations --> API
  API -- SSE --> U
```

## 3. Components

| Module | Responsibility |
|---|---|
| `docassist/config.py` | Typed settings (pydantic-settings, env prefix `DOCASSIST_`) |
| `docassist/api/` | FastAPI app and routes, request validation, SSE streaming |
| `docassist/ingest/` | Per-page PDF text extraction, chunking, ingestion pipeline |
| `docassist/retrieval/` | Embedder wrapper, vector search, full-text search, rank fusion |
| `docassist/llm/` | Anthropic client factory, prompt, answer streaming, citation mapping, refusal handling |
| `docassist/db/` | SQLAlchemy models, async session, Alembic migrations |
| `docassist/web/` | Static `index.html` + `app.js` (no build step) |
| `evals/` | Question set, runner, LLM judge, report |
| `scripts/` | Sample-corpus generator |

Dependencies point inward: `api` → `ingest` / `retrieval` / `llm` → `db`. `llm` and `retrieval` never import FastAPI, so tests and evals call them directly.

## 4. Data model

```
documents  id uuid PK · filename · title (PDF metadata) · sha256 UNIQUE · page_count
           · status (processing|ready|failed) CHECK · error · created_at
chunks     id bigint identity PK · document_id FK → documents ON DELETE CASCADE · ordinal · page
           · text · tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED · embedding vector(768)
indexes    HNSW (embedding vector_cosine_ops) · GIN (tsv) · UNIQUE (document_id, ordinal) · (document_id)
```

The vector dimension is fixed by the embedding model (ADR-3); it is frozen in the migration, and a unit test checks it against the configured model. The `simple` text-search configuration does no stemming but treats English, Russian and Uzbek identically; vector search covers morphology.

## 5. Ingestion

1. **Validate (request):** PDF signature and ≤ 20 MB; anything else gets 415 or 413 before the database is touched. Compute sha256.
   - A known hash returns the existing document (HTTP 200).
   - A known hash whose processing failed is retried (HTTP 202).
2. **Store** the file as `uploads/{sha256}.pdf` and insert the document as `processing`; a background task does the rest.
3. **Extract** text per page with pypdf (≤ 300 pages).
   - A PDF where no page has text is marked `failed` with "No text layer (scanned PDF). OCR is not supported in this demo."
   - Lines repeating at the top or bottom of most pages (headers and footers; digits ignored) are removed, so boilerplate doesn't pollute chunks.
4. **Chunk** within each page (ADR-8): whole sentences up to 120 words, with the last ~20 words of sentences repeated at the start of the next chunk.
5. **Embed** on CPU (fastembed, ONNX). Passages use the model's document format with the PDF title: `title: {title} | text: {chunk}`. Insert all chunks and set `ready` in one transaction.

Failure handling:
- `PdfError` messages are shown to the user.
- Any other error is logged with its traceback and stored as a generic message.
- On startup, documents left `processing` by a restart are marked `failed`, so a re-upload retries them. This assumes a single app instance.

## 6. Retrieval

- **Semantic ranking:** embed the query in the model's query format (`task: search result | query: {q}`) and take the top 20 by cosine distance (HNSW).
- **Exact-identifier boost (ADR-9):** the query's identifiers are:
  - codes with digits: `4.2`, `07/2026`, `A-12`, `N95`;
  - all-caps abbreviations: `ОПТГ`, `VAT`.

  Plain numbers ("20 minutes") don't count. Chunks containing every identifier as a whole lexeme (GIN on `tsv`) move to the top, in semantic order. An identifier matching more than 4 chunks isn't selective and is ignored.
- **Result:** the top 8 chunks. Overlapping chunks of the same page are all kept.
- `GET /search?q=&k=&mode=hybrid|vector|text` shows exactly what the assistant would use as sources. `text` is plain full-text search: an OR query without stopwords, with prefix matching for long words. It is kept for comparison and debugging.
- Retrieval quality is measured by `evals/retrieval.py` against the real database and model. Results are in `evals/results/retrieval.md`.
- There is no hard relevance cut-off. Claude decides "not found" from the sources; the best score is only logged.

## 7. Claude request (answer route)

- **Client:** `AsyncAnthropic` built with explicit `api_key` and `base_url`, so ANTHROPIC_* environment variables are ignored. Requests go through `client.beta.messages.stream`, because the refusal fallback is a beta feature.
- **Model and settings:** `claude-opus-5-5` with `output_config.effort = "medium"` (tuned by eval). `thinking` is omitted: on Opus 5.5 it is always adaptive, and `disabled` would be a 400. `max_tokens = 8000` covers thinking plus the answer.
- **`system`:** a fixed prompt (`llm/prompt.py`). It tells the model to:
  - answer only from the search results and cite them;
  - say "not found" briefly instead of guessing;
  - reply in the question's language and keep amounts and codes verbatim;
  - treat document text as data, not instructions;
  - answer only what was asked, stating each fact once.
- **`messages[0].content`:** one `search_result` block per **page** (ADR-10), then `Question: …` as a text block. Each block has:
  - `source = "doc:{document_id}#page={page}"`;
  - `title = "{filename}, p. {page}"`;
  - `content`: one text block per sentence. The page's retrieved chunks are merged in document order, with the overlapping sentences removed;
  - `citations: {enabled: true}`.
- **Response:** `POST /ask {question, k?, stream?}` returns server-sent events:

  | Event | Meaning |
  |---|---|
  | `sources` | Always first: the pages sent to Claude |
  | `text` | `{block, text}` |
  | `citation` | `{block, source, document_id, filename, page, cited_text}`, mapped from `search_result_index` |
  | `reset` | A `fallback` content block arrived: the requested model declined, and the fallback model starts over. The client discards partial text |
  | `done` | `{stop_reason, model, usage, cost_usd}` |
  | `error` | `{message}`: safe to show. Covers a missing key, a rejected key, and upstream outages |

  - With `stream: false`, the same events are collected into one JSON object with `answer`, `blocks` and their citations.
  - On `stop_reason == "refusal"` the client shows a clear message. On `"max_tokens"`, the partial answer is shown with a notice.
- **Restatement filter (ADR-11):** a text block whose `content_block_start` carries `citations: []` is held until `content_block_stop`. It is then sent whole, or only its citations are sent. Its text is dropped when all of these hold:
  - it is a near-verbatim quote of its cited source (≥ 80% of its content tokens);
  - it starts a new sentence;
  - the previous sentence was written without citations;
  - the two share most of their content tokens (≥ 50% of the shorter, numbers normalized).

  Uncited text still streams token by token. The UI moves the orphaned marker to the end of the previous sentence.
- **Logging:** each answer logs model, stop reason, source count, tokens and estimated cost. Question and answer text are never logged.
- **Web UI** (`src/docassist/web/`, served at `/`): vanilla HTML, CSS and JS with no build step (ADR-6).
  - Documents panel: upload by button or drag and drop, status polling while processing, delete. A ready document's name opens its PDF.
  - Chat: reads the `/ask` event stream with `fetch` (EventSource can't POST). Text is HTML-escaped first, then a minimal Markdown renderer runs (paragraphs, lists, bold).
  - Citations become numbered markers, one number per cited page. Each marker links to `/documents/{id}/file#page=N`, so the browser's PDF viewer opens at that page. A Sources list under the answer shows the quoted text.
  - The answer footer shows model, tokens, cost and pages searched.
  - Styling: light and dark themes via `prefers-color-scheme`; one column under 800 px.
  - Security: a strict CSP (`default-src 'self'`, no inline script or style, checked by a test) on the UI and static files.
- **Known limitation:** citation blocks are sentences. A table has no sentence punctuation, so a whole table chunk is one block, and its `cited_text` is the full table (for example the price list). The page is still exact. Fix later: keep line breaks in chunk text and split table-like units into rows.
- No `output_config.format` on this route: citations and structured outputs are incompatible.
- **Single-turn by design:** every question is independent, so there is no conversation state to keep consistent.

## 8. Evaluation

`evals/questions.yaml` holds 30 items over the generated sample corpus. Each evidence quote is verified against its page by a unit test.

| Group | Count | What's tested |
|---|---|---|
| In scope | 25 | Question, reference answer, evidence file and page. 6 are cross-lingual (an Uzbek or Russian question about an English document, or the reverse), and 2 rely on exact identifiers (`ОПТГ`, clause `4.2`) |
| Out of scope | 4 | Includes a general-knowledge question the model knows but must not answer from memory |
| Injection | 1 | A planted instruction in the supplier agreement; the answer must not follow it |

Two runners:
- `evals/retrieval.py`: retrieval only, no API cost (section 6).
- `evals/run.py`: end to end. Each question goes through retrieval and `/ask` logic. A separate Claude call grades the answer with structured output `Verdict{reason, correct, grounded, language_matches}` at effort `low`, and sees the same excerpts the assistant saw. Citation accuracy is checked without the judge. Concurrency is 3, with up to 6 retries for rate limits.

| Metric | How | Target |
|---|---|---|
| Answer correctness | Judge: core facts match the reference, no contradiction (in scope) | ≥ 90% |
| Citation accuracy | At least one citation is on an evidence page (in scope) | ≥ 90% |
| Out of scope handled | Judge: says "not found", no answer from general knowledge | 100% |
| Injection resisted | Judge: doesn't claim treatments are free | 100% |
| Grounded | Judge: every factual claim is supported by the excerpts (all answers) | ≥ 95% |
| Language match | Judge: reply in the question's language (all answers) | ≥ 95% |
| No repetition | Deterministic: no two sentences or clauses share ≥ 75% of the shorter one's words (all answers) | ≥ 95% |
| Retrieval hit, cost, latency | Evidence page among sources; `usage` × price; time to first word and full answer | reported |

`evals/run.py` writes `evals/results/latest.md` (committed; its summary goes into the README) and `latest.jsonl` with raw answers (git-ignored). It exits 1 when a target is missed.

Caveat: the judge is the same model family as the assistant. Its verdicts are checked by reading the per-question table, and every failure carries the judge's reason.

## 9. Security and privacy

- Uploaded files are stored in a volume under their sha256 name; displayed filenames are sanitized; size and page limits are enforced.
- Document text is untrusted. It reaches Claude only as `search_result` content, never as instructions. The model has no tools, so injected text cannot trigger actions.
- Secrets come from the environment only, and the API key is never logged. Logs hold request id, timings and token usage, never document text.
- CORS is limited to the same origin.

## 10. Operations

- `docker compose up -d --build` starts two services:
  - `db`: `pgvector/pgvector:pg17`, named volume, healthcheck;
  - `app`: Python 3.13 slim. The embedding model is downloaded at build time, so the container starts offline.
- Configuration comes from `.env`; the template is `.env.example`.
- Cost is logged per request: `usage` (input, output and cache tokens) × the price table in settings.
- CI (GitHub Actions) runs ruff and pytest on every push.

## 11. Decisions

| ID | Decision | Why / trade-off |
|---|---|---|
| ADR-1 | PostgreSQL + pgvector instead of a dedicated vector DB | One database for metadata, vectors and full-text; what clients already run (Supabase, RDS) |
| ADR-2 | Local embeddings (fastembed, ONNX) instead of an embeddings API | No second paid key, works offline, data stays local. Trade-off: bigger image, CPU time at ingestion |
| ADR-3 | Embedding model: `google/embeddinggemma-300m` (768 dims), chosen in step 3 with `evals/embedding_benchmark.py`. Over 23 questions it found the right page first in 91% of cases (MRR 0.93); potion-multilingual got 0.86, Qwen3-0.6B-Q 0.83, MiniLM-L12 0.71. Results are in `evals/results/embedding_benchmark.md` | The best ranking at 23 ms per query on CPU. Trade-offs: a 1.2 GB model in the image, about 0.2 s per chunk at ingestion, and the Gemma Terms of Use licence. `minishlab/potion-multilingual-128M` (MIT, 0.5 GB, 0.4 ms per query) is the fallback if a client can't accept those terms. Switching models needs a new migration and re-ingestion |
| ADR-4 | `search_result` blocks with native citations instead of quotes requested in the prompt | Exact quoted spans, `cited_text` not billed as output, no parsing |
| ADR-5 | pypdf (BSD) instead of PyMuPDF (AGPL) | Licence-safe for client code. Trade-off: weaker layout handling |
| ADR-6 | Vanilla JS UI instead of React | No build step; the demo is about the backend |
| ADR-8 | Chunks never cross page boundaries (120 words, ~20-word overlap) instead of ~350-word chunks spanning pages | Every chunk has exactly one page, so citations point to one page. Smaller chunks also rank more precisely, and 8 of them still cost under 1.5k input tokens. Trade-off: a sentence broken across a page break is split in two |
| ADR-9 | Semantic ranking with an exact-identifier boost instead of Reciprocal Rank Fusion of vector and full-text results | Measured on 25 questions, MRR: RRF (k = 60) 0.83; weighted RRF up to 0.86 (text weight 0.3–0.5, depth 5); vector only 0.94; vector + boost 0.94. With k = 60, adjacent vector ranks differ by about 0.0003, so any full-text contribution reorders results. Cross-lingual questions then lose to chunks that only share words like "price". The boost keeps the semantic order and guarantees the case embeddings handle worst: exact references. Trade-off: no keyword help for terms that are neither codes nor abbreviations |
| ADR-10 | One `search_result` per page with sentence-level content blocks, instead of one per chunk | Citations point to a page, and overlapping chunks of a page would otherwise be sent twice. The text block is the smallest unit Claude can cite, so sentence blocks make `cited_text` a sentence rather than a 120-word chunk |
| ADR-11 | Drop cited quotes that restate the previous sentence in code, not by prompt | With citations on, the model sometimes writes a fact and then repeats it as a cited verbatim quote; the first eval run showed it in 2 of 30 answers. Two prompt rewrites didn't fix it; one made it worse (3 of 3 runs, every fact doubled), so it was reverted. The filter is deterministic, unit-tested on recorded API event sequences, and keeps every citation. Trade-off: cited blocks arrive whole instead of token by token, and the rule can miss paraphrases that share few words. `evals/run.py` measures what remains (the "no repetition" metric) |
| ADR-7 | Server-side refusal fallback (`fallbacks: "default"`, beta header `server-side-fallback-2026-07-01`) on, behind a setting | Requests are single-turn, so a fallback has no history side effects. Trade-off: a beta dependency, which the setting turns off |

## 12. Extensions (offer as add-ons)

OCR via Claude vision on page images · DOCX/XLSX · authentication and per-team spaces · Telegram or Slack front end · conversation memory · Google Drive / Notion sync · hosted deployment.
