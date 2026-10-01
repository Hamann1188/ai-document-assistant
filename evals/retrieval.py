"""Retrieval eval: does search put the evidence page among the top-k chunks?

Runs the real pipeline (PostgreSQL, the configured embedding model) over the
in-scope questions of evals/questions.yaml, for each search mode. Sample documents
missing from the database are ingested first. No API key is needed.

Needs the database from docker compose. Run:
    uv run python -m evals.retrieval

Exit code 1 if hybrid recall@8 is below the target.
"""

import asyncio
import sys
from pathlib import Path

import yaml

from docassist.config import EMBEDDING_MODEL, Settings
from docassist.db.session import create_engine, create_session_factory
from docassist.ingest.pipeline import process_document, register_upload
from docassist.retrieval.embedder import FastEmbedder
from docassist.retrieval.search import SearchMode, search

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results" / "retrieval.md"
KS = (1, 3, 5, 8)
TARGET_RECALL_AT_8 = 0.9


async def ensure_sample_corpus(session_factory, embedder, settings: Settings) -> None:
    for path in sorted((ROOT / "sample_docs").glob("*.pdf")):
        async with session_factory() as session:
            registration = await register_upload(session, settings, path.name, path.read_bytes())
        if registration.needs_processing:
            print(f"ingesting {path.name}", flush=True)
            await process_document(session_factory, embedder, settings, registration.document.id)


async def evaluate(session_factory, embedder, questions, mode: SearchMode) -> dict:
    hits_at = dict.fromkeys(KS, 0)
    reciprocal_ranks, misses = [], []
    for item in questions:
        expected = {(ev["file"], ev["page"]) for ev in item["evidence"]}
        async with session_factory() as session:
            hits = await search(session, embedder, item["question"], limit=max(KS), mode=mode)
        rank = next(
            (i for i, hit in enumerate(hits, start=1) if (hit.filename, hit.page) in expected),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        for k in KS:
            hits_at[k] += bool(rank and rank <= k)
        if not rank or rank > 3:
            misses.append(f"{item['id']} ({f'rank {rank}' if rank else 'not in top 8'})")
    n = len(questions)
    return {
        "mode": mode.value,
        **{f"R@{k}": hits_at[k] / n for k in KS},
        "MRR": sum(reciprocal_ranks) / n,
        "misses": misses,
    }


async def main() -> int:
    settings = Settings()
    questions = [
        q
        for q in yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))
        if q["kind"] == "in_scope"
    ]
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    embedder = FastEmbedder(EMBEDDING_MODEL, cache_dir=ROOT / ".cache" / "fastembed")
    try:
        await ensure_sample_corpus(session_factory, embedder, settings)
        results = [
            await evaluate(session_factory, embedder, questions, mode) for mode in SearchMode
        ]
    finally:
        await engine.dispose()

    header = "| Mode | " + " | ".join(f"R@{k}" for k in KS) + " | MRR |"
    lines = [
        f"Retrieval over the sample corpus: {len(questions)} in-scope questions, "
        f"embedding model `{EMBEDDING_MODEL}`.",
        "A hit is a chunk from an evidence page.",
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
    ]
    for r in results:
        scores = " | ".join(f"{r[f'R@{k}']:.2f}" for k in KS)
        lines.append(f"| {r['mode']} | {scores} | {r['MRR']:.2f} |")
    lines += ["", "Ranked below 3:", ""]
    lines += [f"- {r['mode']}: {', '.join(r['misses']) or 'none'}" for r in results]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    hybrid = next(r for r in results if r["mode"] == SearchMode.HYBRID)
    passed = hybrid["R@8"] >= TARGET_RECALL_AT_8
    print(f"\nhybrid R@8 = {hybrid['R@8']:.2f} (target {TARGET_RECALL_AT_8}): ", end="")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
