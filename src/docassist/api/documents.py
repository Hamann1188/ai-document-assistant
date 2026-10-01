import uuid
from datetime import datetime

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from docassist.db.models import Document
from docassist.ingest.pipeline import process_document, register_upload, stored_path

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    title: str | None
    status: str
    error: str | None
    page_count: int | None
    created_at: datetime


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "The same file was already uploaded"}},
)
async def upload_document(
    request: Request, file: UploadFile, background_tasks: BackgroundTasks
) -> JSONResponse:
    """Upload a PDF. Processing runs in the background; poll GET /documents/{id}."""
    settings = request.app.state.settings
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {settings.max_upload_mb} MB."
        )
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported.")

    async with request.app.state.session_factory() as session:
        registration = await register_upload(session, settings, file.filename, data)
    if registration.needs_processing:
        background_tasks.add_task(
            process_document,
            request.app.state.session_factory,
            request.app.state.embedder,
            settings,
            registration.document.id,
        )
    return JSONResponse(
        DocumentOut.model_validate(registration.document).model_dump(mode="json"),
        status_code=status.HTTP_202_ACCEPTED
        if registration.needs_processing
        else status.HTTP_200_OK,
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(request: Request) -> list[Document]:
    async with request.app.state.session_factory() as session:
        result = await session.scalars(select(Document).order_by(Document.created_at.desc()))
        return list(result)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(request: Request, document_id: uuid.UUID) -> Document:
    async with request.app.state.session_factory() as session:
        document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return document


@router.get(
    "/{document_id}/file",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def get_document_file(request: Request, document_id: uuid.UUID) -> FileResponse:
    """The original PDF, shown inline. Append `#page=N` to open it at a page."""
    async with request.app.state.session_factory() as session:
        document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    path = stored_path(request.app.state.settings, document.sha256)
    if not await anyio.Path(path).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The stored file is missing.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(request: Request, document_id: uuid.UUID) -> None:
    async with request.app.state.session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
        await session.delete(document)  # chunks go with it (ON DELETE CASCADE)
        await session.commit()
    await anyio.Path(stored_path(request.app.state.settings, document.sha256)).unlink(
        missing_ok=True
    )
