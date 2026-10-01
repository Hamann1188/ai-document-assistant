import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from docassist.llm.answer import collect_answer, stream_answer
from docassist.llm.prompt import group_sources
from docassist.retrieval.search import search

router = APIRouter(tags=["answers"])


class AskIn(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    k: Annotated[int | None, Field(ge=1, le=20, description="Chunks to retrieve")] = None
    stream: bool = True

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is empty")
        return value.strip()


def to_sse(event: dict) -> str:
    payload = {k: v for k, v in event.items() if k != "type"}
    return f"event: {event['type']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/ask",
    responses={
        200: {
            "description": "Server-sent events (sources, text, citation, reset, done, error), "
            "or one JSON object when `stream` is false",
            "content": {"text/event-stream": {}, "application/json": {}},
        }
    },
)
async def ask(request: Request, body: AskIn):
    """Answer a question from the uploaded documents, citing file and page."""
    state = request.app.state
    async with state.session_factory() as session:
        hits = await search(
            session, state.embedder, body.question, limit=body.k or state.settings.retrieval_k
        )
    events = stream_answer(state.llm_client, state.settings, body.question, group_sources(hits))

    if not body.stream:
        return JSONResponse((await collect_answer(events)).to_dict())

    async def sse() -> AsyncIterator[str]:
        async for event in events:
            yield to_sse(event)

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
