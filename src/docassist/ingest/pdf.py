import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfError(ValueError):
    """A PDF that can't be ingested. The message is safe to show to the user."""


@dataclass(frozen=True)
class ExtractedPdf:
    title: str | None
    pages: list[str]  # pages[0] is page 1


def extract_pdf(data: bytes, max_pages: int) -> ExtractedPdf:
    """Extract text per page, with running headers and footers removed."""
    if not data.startswith(b"%PDF-"):
        raise PdfError("Not a PDF file.")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise PdfError("Password-protected PDFs are not supported.")
        if len(reader.pages) > max_pages:
            raise PdfError(f"The PDF has {len(reader.pages)} pages; the limit is {max_pages}.")
        pages = [page.extract_text() or "" for page in reader.pages]
        title = reader.metadata.title if reader.metadata else None
    except PdfReadError as exc:
        raise PdfError("The PDF is damaged and can't be read.") from exc

    if not any(page.strip() for page in pages):
        raise PdfError("No text layer (scanned PDF). OCR is not supported in this demo.")
    return ExtractedPdf(title=(title or "").strip() or None, pages=strip_running_lines(pages))


_DIGITS = re.compile(r"\d+")


def strip_running_lines(pages: list[str], edge_lines: int = 2) -> list[str]:
    """Drop header/footer lines that repeat (ignoring numbers) on most pages.

    Only the first and last `edge_lines` lines of each page are candidates, so
    repeated lines in the body (e.g. table headers) are kept.
    """
    if len(pages) < 3:
        return pages

    def key(line: str) -> str:
        return _DIGITS.sub("#", line.strip())

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.splitlines()
        edges = {key(line) for line in lines[:edge_lines] + lines[-edge_lines:] if line.strip()}
        counts.update(edges)
    threshold = max(2, round(len(pages) * 0.6))
    running = {line for line, count in counts.items() if count >= threshold}

    cleaned = []
    for page in pages:
        lines = page.splitlines()
        head, body, tail = lines[:edge_lines], lines[edge_lines:-edge_lines], lines[-edge_lines:]
        if len(lines) <= 2 * edge_lines:
            head, body, tail = lines, [], []
        kept = [line for line in head if key(line) not in running] + body
        kept += [line for line in tail if key(line) not in running]
        cleaned.append("\n".join(kept).strip())
    return cleaned
