from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from docassist.api.main import create_app
from docassist.db.models import EMBEDDING_DIM
from docassist.ingest.pipeline import INTERNAL_ERROR

SAMPLE_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"
SAMPLE_PAGES = {
    "patient-handbook.pdf": 5,
    "price-list-ru.pdf": 3,
    "employee-handbook.pdf": 4,
    "supplier-agreement.pdf": 4,
}


def upload(client, name: str, data: bytes | None = None):
    data = data if data is not None else (SAMPLE_DOCS / name).read_bytes()
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_all_sample_documents_ingest(client, sql):
    for name, pages in SAMPLE_PAGES.items():
        response = upload(client, name)
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "processing"
        # TestClient runs background tasks before returning, so processing is done.
        document = client.get(f"/documents/{response.json()['id']}").json()
        assert document["status"] == "ready", document
        assert document["page_count"] == pages

        rows = sql(
            "SELECT ordinal, page FROM chunks WHERE document_id = :id ORDER BY ordinal",
            id=document["id"],
        )
        assert [r.ordinal for r in rows] == list(range(len(rows)))
        assert {r.page for r in rows} == set(range(1, pages + 1))

    assert len(client.get("/documents").json()) == 4


def test_chunks_are_searchable_by_full_text_and_vector(client, sql):
    document = upload(client, "patient-handbook.pdf").json()
    hits = sql(
        "SELECT page FROM chunks WHERE tsv @@ websearch_to_tsquery('simple', 'parking courtyard')"
    )
    assert [h.page for h in hits] == [1]
    (count,) = sql(
        "SELECT count(*) FROM chunks WHERE document_id = :id AND vector_dims(embedding) = :dim",
        id=document["id"],
        dim=EMBEDDING_DIM,
    )[0]
    assert count > 0
    assert not sql("SELECT 1 FROM chunks WHERE text LIKE '%Fictional demo document%'")


def test_same_file_is_stored_once(client, embedder):
    first = upload(client, "price-list-ru.pdf")
    second = upload(client, "price-list-ru.pdf")
    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json()["id"] == first.json()["id"]
    assert embedder.calls == 1


def test_pdf_without_text_fails_with_reason_and_can_be_retried(client, embedder):
    response = upload(client, "scan.pdf", blank_pdf())
    document = client.get(f"/documents/{response.json()['id']}").json()
    assert document["status"] == "failed"
    assert "No text layer" in document["error"]

    retry = upload(client, "scan.pdf", blank_pdf())
    assert retry.status_code == 202
    assert retry.json()["id"] == document["id"]
    assert embedder.calls == 0


def test_unexpected_error_marks_document_failed_without_leaking_details(
    settings, fake_embedder_cls, sql
):
    broken = fake_embedder_cls(EMBEDDING_DIM, fail=True)
    with TestClient(create_app(settings, embedder_factory=lambda _: broken)) as client:
        response = upload(client, "employee-handbook.pdf")
        document = client.get(f"/documents/{response.json()['id']}").json()
    assert document["status"] == "failed"
    assert document["error"] == INTERNAL_ERROR
    assert not sql("SELECT 1 FROM chunks")


def test_file_endpoint_serves_the_original_pdf_inline(client, settings):
    original = (SAMPLE_DOCS / "price-list-ru.pdf").read_bytes()
    document = upload(client, "price-list-ru.pdf").json()

    response = client.get(f"/documents/{document['id']}/file")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert 'filename="price-list-ru.pdf"' in response.headers["content-disposition"]
    assert response.content == original

    for stored in settings.upload_dir.glob("*.pdf"):
        stored.unlink()
    assert client.get(f"/documents/{document['id']}/file").status_code == 404


def test_file_endpoint_404_for_unknown_document(client):
    response = client.get("/documents/00000000-0000-0000-0000-000000000000/file")
    assert response.status_code == 404


def test_delete_removes_document_chunks_and_file(client, settings, sql):
    document = upload(client, "supplier-agreement.pdf").json()
    stored = list(settings.upload_dir.glob("*.pdf"))
    assert len(stored) == 1

    assert client.delete(f"/documents/{document['id']}").status_code == 204
    assert client.get(f"/documents/{document['id']}").status_code == 404
    assert not sql("SELECT 1 FROM chunks")
    assert not stored[0].exists()
    assert client.delete(f"/documents/{document['id']}").status_code == 404


def test_startup_fails_documents_interrupted_by_a_restart(settings, embedder, sql):
    sql(
        "INSERT INTO documents (id, filename, sha256, status) "
        "VALUES (gen_random_uuid(), 'x.pdf', repeat('a', 64), 'processing')"
    )
    with TestClient(create_app(settings, embedder_factory=lambda _: embedder)) as client:
        (document,) = client.get("/documents").json()
    assert document["status"] == "failed"
    assert document["error"] == INTERNAL_ERROR
