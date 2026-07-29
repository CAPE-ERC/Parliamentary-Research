"""Per-page text extraction from Hansard debate PDFs."""

from pathlib import Path

import pdfplumber


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the extracted text of each page in the PDF, in order."""
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]
