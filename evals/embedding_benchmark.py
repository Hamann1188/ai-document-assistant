"""Compare embedding models on vector-only retrieval over the sample corpus.

For each in-scope question in evals/questions.yaml, a hit means a retrieved chunk
comes from a page listed in the question's evidence. No database or API key is
needed; models download on first run (about 3 GB for all of them).

Run: uv run python -m evals.embedding_benchmark [model ...]
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

from docassist.ingest.chunker import chunk_pages
from docassist.ingest.pdf import extract_pdf
from docassist.retrieval.embedder import MODELS, FastEmbedder

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "fastembed"
RESULTS = ROOT / "evals" / "results" / "embedding_benchmark.md"
MAX_WORDS, OVERLAP_WORDS = 120, 20
KS = (1, 3, 5, 8)


def load_corpus() -> tuple[list[tuple[str, int]], list[tuple[str, str]]]:
    keys, passages = [], []
    for path in sorted((ROOT / "sample_docs").glob("*.pdf")):
        pdf = extract_pdf(path.read_bytes(), max_pages=300)
        for chunk in chunk_pages(pdf.pages, MAX_WORDS, OVERLAP_WORDS):
            keys.append((path.name, chunk.page))
            passages.append((pdf.title or path.stem, chunk.text))
    return keys, passages


def evaluate(model_name: str, keys, passages, questions) -> dict:
    embedder = FastEmbedder(model_name, cache_dir=CACHE)
    started = time.perf_counter()
    embedder.embed_query("warm-up")  # loads (and on first run downloads) the model
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    matrix = np.array(embedder.embed_passages(passages))
    index_s = time.perf_counter() - started
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    hits = dict.fromkeys(KS, 0)
    reciprocal_ranks, misses = [], []
    started = time.perf_counter()
    for item in questions:
        query = np.array(embedder.embed_query(item["question"]))
        ranking = np.argsort(-(matrix @ (query / np.linalg.norm(query))))
        expected = {(ev["file"], ev["page"]) for ev in item["evidence"]}
        rank = next(i for i, idx in enumerate(ranking, start=1) if keys[idx] in expected)
        reciprocal_ranks.append(1 / rank)
        for k in KS:
            hits[k] += rank <= k
        if rank > 3:
            misses.append(f"{item['id']} (rank {rank})")
    query_ms = (time.perf_counter() - started) / len(questions) * 1000

    n = len(questions)
    return {
        "model": model_name,
        "dim": embedder.dim,
        **{f"R@{k}": hits[k] / n for k in KS},
        "MRR": sum(reciprocal_ranks) / n,
        "load_s": load_s,
        "index_s": index_s,
        "query_ms": query_ms,
        "misses": misses,
    }


def main() -> None:
    models = sys.argv[1:] or list(MODELS)
    questions = [
        q
        for q in yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))
        if q["kind"] == "in_scope"
    ]
    keys, passages = load_corpus()
    print(f"{len(passages)} chunks, {len(questions)} in-scope questions", flush=True)

    rows = []
    for model in models:
        result = evaluate(model, keys, passages, questions)
        rows.append(result)
        print(result, flush=True)

    header = "| Model | Dim | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | Query ms |"
    lines = [
        f"Vector-only retrieval: {len(passages)} chunks ({MAX_WORDS} words max, "
        f"{OVERLAP_WORDS} overlap), {len(questions)} in-scope questions.",
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
    ]
    for r in sorted(rows, key=lambda r: -r["MRR"]):
        scores = " | ".join(f"{r[f'R@{k}']:.2f}" for k in KS)
        lines.append(
            f"| {r['model']} | {r['dim']} | {scores} | {r['MRR']:.2f} | {r['query_ms']:.0f} |"
        )
    lines += ["", "Questions ranked below 3:", ""]
    lines += [f"- {r['model']}: {', '.join(r['misses']) or 'none'}" for r in rows]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
