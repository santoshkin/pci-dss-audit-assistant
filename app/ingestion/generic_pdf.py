"""Plain single-column PDF text extraction for documents that are not the
PCI DSS Requirements/Testing Procedures table template - FAQ and Guidance
PDFs (`data/documents/faq/`, `data/documents/guidance/`). See
ARCHITECTURE.md section 14 for why these use a lighter-weight path than
`app/ingestion`'s coordinate-based parser: arbitrary per-document layout,
high volume, and the user asked for an accuracy/speed balance here rather
than the Standard's maximum-precision treatment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)*\.?x?)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b([A-Za-z]+ \d{4})\b")


class GenericPdfParser:
    def __init__(self, path: Path) -> None:
        self.path = path

    def extract_pages(self) -> list[str]:
        with pdfplumber.open(self.path) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]

    def guess_title(self, pages: list[str]) -> str:
        for page_text in pages:
            for line in page_text.splitlines():
                if line.strip():
                    return line.strip()
        return self.path.stem

    def guess_version(self, pages: list[str]) -> str:
        for page_text in pages[:2]:
            match = _VERSION_RE.search(page_text)
            if match:
                return match.group(1)
        return "unknown"

    def guess_date(self, pages: list[str]) -> str:
        for page_text in pages[:2]:
            match = _DATE_RE.search(page_text)
            if match:
                return match.group(1)
        return "unknown"
