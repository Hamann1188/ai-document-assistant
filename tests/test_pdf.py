from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from docassist.ingest.pdf import PdfError, extract_pdf, strip_running_lines

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "sample_docs"


def test_extracts_pages_title_and_drops_running_footer():
    pdf = extract_pdf((SAMPLE_DOCS / "patient-handbook.pdf").read_bytes(), max_pages=300)
    assert pdf.title == "Registan Smile Clinic - Patient Handbook"
    assert len(pdf.pages) == 5
    assert "Saturday: 10:00 - 16:00" in pdf.pages[0]
    assert not any("Fictional demo document" in page for page in pdf.pages)


def test_extracts_cyrillic_text():
    pdf = extract_pdf((SAMPLE_DOCS / "price-list-ru.pdf").read_bytes(), max_pages=300)
    assert "Коронка из диоксида циркония" in pdf.pages[2]


def test_rejects_non_pdf():
    with pytest.raises(PdfError, match="Not a PDF"):
        extract_pdf(b"hello", max_pages=10)


def test_rejects_pdf_over_page_limit():
    data = (SAMPLE_DOCS / "patient-handbook.pdf").read_bytes()
    with pytest.raises(PdfError, match="5 pages; the limit is 3"):
        extract_pdf(data, max_pages=3)


def test_rejects_pdf_without_text_layer():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(PdfError, match="No text layer"):
        extract_pdf(buffer.getvalue(), max_pages=10)


def test_rejects_damaged_pdf():
    with pytest.raises(PdfError, match="damaged"):
        extract_pdf(b"%PDF-1.7\n garbage without structure", max_pages=10)


def test_strip_running_lines_keeps_body_lines_that_repeat():
    words = ["alpha", "beta", "gamma", "delta"]
    pages = [
        f"ACME report | Page {n}\nIntro {w}\nPrice table\nItem 1 100\nClosing {w}\nConfidential"
        for n, w in enumerate(words, start=1)
    ]
    assert strip_running_lines(pages) == [
        f"Intro {w}\nPrice table\nItem 1 100\nClosing {w}" for w in words
    ]


def test_strip_running_lines_leaves_short_documents_alone():
    pages = ["Header\nText one", "Header\nText two"]
    assert strip_running_lines(pages) == pages
