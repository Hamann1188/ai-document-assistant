import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from docassist.retrieval.search import SearchMode, search

router = APIRouter(tags=["search"])


class SearchHitOut(BaseModel):
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    title: str | None
    page: int
    text: str
    vector_rank: int | None
    text_rank: int | None
    exact_match: bool


class SearchOut(BaseModel):
    query: str
    mode: SearchMode
    hits: list[SearchHitOut]


@router.get("/search", response_model=SearchOut)
async def search_chunks(
    request: Request,
    q: Annotated[str, Query(max_length=1000, description="Question or keywords")],
    k: Annotated[int, Query(ge=1, le=20, description="Number of chunks to return")] = 8,
    mode: SearchMode = SearchMode.HYBRID,
) -> SearchOut:
    """Show the chunks the assistant would use as sources for a question."""
    query = q.strip()
    if not query:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Query is empty.")
    async with request.app.state.session_factory() as session:
        hits = await search(session, request.app.state.embedder, query, limit=k, mode=mode)
    return SearchOut(query=query, mode=mode, hits=[SearchHitOut(**vars(hit)) for hit in hits])
