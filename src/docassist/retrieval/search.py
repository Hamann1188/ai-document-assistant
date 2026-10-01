"""Retrieval over chunks: semantic (pgvector) ranking with exact-identifier boosting.

Hybrid mode ranks by embedding similarity, then moves chunks that contain every
exact identifier from the question (clause numbers, codes, abbreviations) to the
top. Embeddings blur such tokens: "4.2" and "4.3" look alike to a vector model.
Rank fusion (RRF) was measured and rejected, see docs/ARCHITECTURE.md (ADR-9).
"""

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import anyio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from docassist.db.models import Chunk, Document, DocumentStatus
from docassist.retrieval.embedder import Embedder

# An identifier that matches more chunks than this isn't selective; ignore it.
MAX_BOOSTED = 4

# Question words and function words, for the text-only mode. The 'simple'
# text-search config keeps every word, and an OR query on them would match
# almost every chunk.
_STOPWORD_LIST = (
    # English
    "a about all an and any are as at be can could do does for from has have how i if "
    "in is it its me my no not of on or our should so than that the their there these "
    "this to us was we what when where which who why will with would you your "
    # Russian
    "а без бы в во вы где да для до если есть же за и из или к как какая какие какой "
    "когда ко ли мне можно мы на не нет но о об от по под при с сколько со то у что "
    "это я ваш ваша ваше ваши "
    # Uzbek (Latin)
    "bilan bu bormi kim mi men nima nechta qachon qancha qanday qayerda siz u uchun va"
)
STOPWORDS = frozenset(_STOPWORD_LIST.split())

_TOKEN = re.compile(r"[\w./-]+")


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    TEXT = "text"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    title: str | None
    page: int
    text: str
    vector_rank: int | None  # position in the semantic ranking
    text_rank: int | None  # position in the full-text ranking
    exact_match: bool  # contains every identifier from the query


def extract_identifiers(query: str) -> list[str]:
    """Tokens that must match exactly: numbered codes and all-caps abbreviations.

    "4.2", "07/2026", "A-12" and "N95" count; plain numbers ("20 minutes") don't,
    because in questions they are usually values, not references. Abbreviations
    need 2+ letters: "ОПТГ", "VAT".
    """
    found: dict[str, None] = {}
    for raw in _TOKEN.findall(query):
        token = raw.strip("./-")
        has_digit = any(c.isdigit() for c in token)
        has_alpha = any(c.isalpha() for c in token)
        is_code = has_digit and (has_alpha or any(c in "./-" for c in token))
        is_abbreviation = not has_digit and len(token) >= 2 and token.isupper()
        if is_code or is_abbreviation:
            found[token] = None
    return list(found)


def build_tsquery(lexemes: Iterable[str]) -> str | None:
    """OR-query over meaningful lexemes, with crude prefix stemming for long words.

    A word of 5+ letters matches by its prefix without the last two letters (at
    least 4): "rescheduled" -> 'reschedul':* finds "reschedule", and "имплантации" ->
    'имплантац':* finds "имплантация". Numbers and codes ("4.2", "07/2026") must
    match exactly.
    """
    terms: dict[str, None] = {}
    for lexeme in lexemes:
        if lexeme in STOPWORDS or (len(lexeme) < 2 and not lexeme.isdigit()):
            continue
        if lexeme.isalpha() and len(lexeme) >= 5:
            terms[f"{_quote(lexeme[: max(4, len(lexeme) - 2)])}:*"] = None
        else:
            terms[_quote(lexeme)] = None
    return " | ".join(terms) or None


def _quote(lexeme: str) -> str:
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


_ROW_COLUMNS = "c.id, c.document_id, d.filename, d.title, c.page, c.text"

_LEXEMES_SQL = text("SELECT tsvector_to_array(to_tsvector('simple', :query))")

_TEXT_SEARCH_SQL = text(
    f"""
    SELECT {_ROW_COLUMNS}
    FROM chunks c
    JOIN documents d ON d.id = c.document_id,
         to_tsquery('simple', :tsquery) AS q
    WHERE d.status = 'ready' AND c.tsv @@ q
    ORDER BY ts_rank(c.tsv, q) DESC, c.id
    LIMIT :limit
    """
)

# Identifiers are matched as whole lexemes with the same parser that built `tsv`,
# so "4.2" doesn't match "14.2" or "4.25".
_IDENTIFIER_SQL = text(
    f"""
    SELECT {_ROW_COLUMNS}
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.status = 'ready'
      AND c.tsv @@ (SELECT string_agg(quote_literal(lexeme), ' & ')::tsquery
                    FROM unnest(tsvector_to_array(to_tsvector('simple', :identifiers))) lexeme)
    ORDER BY c.id
    LIMIT :limit
    """
)


async def search(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    limit: int = 8,
    candidates: int = 20,
    mode: SearchMode = SearchMode.HYBRID,
) -> list[SearchHit]:
    rows: dict[int, object] = {}
    vector_ids: list[int] = []
    text_ids: list[int] = []
    exact_ids: list[int] = []

    if mode is not SearchMode.TEXT:
        vector = await anyio.to_thread.run_sync(embedder.embed_query, query)
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.filename,
                Document.title,
                Chunk.page,
                Chunk.text,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.READY)
            .order_by(Chunk.embedding.cosine_distance(vector))
            .limit(candidates)
        )
        for row in await session.execute(stmt):
            vector_ids.append(row.id)
            rows[row.id] = row

    if mode is SearchMode.TEXT:
        lexemes = await session.scalar(_LEXEMES_SQL, {"query": query})
        tsquery = build_tsquery(lexemes or [])
        if tsquery:
            result = await session.execute(
                _TEXT_SEARCH_SQL, {"tsquery": tsquery, "limit": candidates}
            )
            for row in result:
                text_ids.append(row.id)
                rows[row.id] = row

    if mode is SearchMode.HYBRID and (identifiers := extract_identifiers(query)):
        result = await session.execute(
            _IDENTIFIER_SQL, {"identifiers": " ".join(identifiers), "limit": MAX_BOOSTED + 1}
        )
        matches = result.all()
        if len(matches) <= MAX_BOOSTED:
            for row in matches:
                exact_ids.append(row.id)
                rows[row.id] = row

    # Exact matches first (in semantic order where ranked), then the semantic ranking.
    vector_rank = {chunk_id: i for i, chunk_id in enumerate(vector_ids, start=1)}
    text_rank = {chunk_id: i for i, chunk_id in enumerate(text_ids, start=1)}
    exact = sorted(exact_ids, key=lambda c: vector_rank.get(c, len(vector_ids) + 1))
    ordered = list(dict.fromkeys([*exact, *vector_ids, *text_ids]))

    return [
        SearchHit(
            chunk_id=chunk_id,
            document_id=rows[chunk_id].document_id,
            filename=rows[chunk_id].filename,
            title=rows[chunk_id].title,
            page=rows[chunk_id].page,
            text=rows[chunk_id].text,
            vector_rank=vector_rank.get(chunk_id),
            text_rank=text_rank.get(chunk_id),
            exact_match=chunk_id in exact_ids,
        )
        for chunk_id in ordered[:limit]
    ]
