"""Structure-lite chunking for FAQ/Guidance PDFs and organizational .docx
documents - lighter-weight than `StructureAwareChunker` by design
(ARCHITECTURE.md section 14): headings are found by a cheap heuristic
(PDF) or read directly from Word styles (.docx) instead of the exhaustively
verified coordinate parsing built for the PCI DSS Standard, since these
documents are higher-volume and arbitrary-format, and the user asked for
an accuracy/speed balance here rather than maximum precision.
"""

from __future__ import annotations

import re

from app.chunking.chunker import Chunk, split_with_overlap
from app.ingestion.docx_parser import DocxSection

# A numbered heading ("2.1 What is...", "3. Overview") is the cheapest
# reliable structure signal across these documents' differing templates -
# confirmed present in both the FAQ and Guidance PDFs checked. A line not
# matching this is just treated as body text; less structure than the
# Standard gets, by design (see module docstring).
_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[.\s]+\S.{2,100}$")


def _looks_like_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line.strip()))


def _sections_from_pages(pages: list[str]) -> list[tuple[str | None, int, int, str]]:
    """Returns (heading, page_start, page_end, text) tuples, page numbers 1-based."""
    sections: list[tuple[str | None, int, int, str]] = []
    heading: str | None = None
    start_page = 1
    lines: list[str] = []

    def flush(end_page: int) -> None:
        if lines:
            sections.append((heading, start_page, end_page, "\n".join(lines)))

    for page_number, page_text in enumerate(pages, start=1):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _looks_like_heading(line):
                flush(page_number)
                heading, start_page, lines = line, page_number, []
                continue
            lines.append(line)
    flush(len(pages) or 1)
    return sections


class GenericChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pdf(
        self,
        *,
        pages: list[str],
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        sections = _sections_from_pages(pages)
        return self._emit(
            sections, document_id, document_title, document_type, version, publication_date, source_file
        )

    def _emit(
        self,
        sections: list[tuple[str | None, int, int, str]],
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, (heading, page_start, page_end, text) in enumerate(sections):
            body = f"{heading}\n\n{text}" if heading else text
            for part_idx, part in enumerate(split_with_overlap(body, self.chunk_size, self.chunk_overlap)):
                chunks.append(
                    Chunk(
                        id=f"{document_id}::section{idx}::{part_idx}",
                        document_id=document_id,
                        document_title=document_title,
                        document_type=document_type,
                        version=version,
                        publication_date=publication_date,
                        page_start=page_start,
                        page_end=page_end,
                        section=heading,
                        requirement_id=None,
                        testing_procedure_ids=(),
                        chunk_type="generic",
                        subsection=None,
                        parent_chunk_id=None,
                        source_file=source_file,
                        text=part,
                    )
                )
        return chunks

    def chunk_text(
        self,
        *,
        text: str,
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for part_idx, part in enumerate(split_with_overlap(text, self.chunk_size, self.chunk_overlap)):
            chunks.append(
                Chunk(
                    id=f"{document_id}::part{part_idx}",
                    document_id=document_id,
                    document_title=document_title,
                    document_type=document_type,
                    version=version,
                    publication_date=publication_date,
                    page_start=0,
                    page_end=0,
                    section=None,
                    requirement_id=None,
                    testing_procedure_ids=(),
                    chunk_type="generic",
                    subsection=None,
                    parent_chunk_id=None,
                    source_file=source_file,
                    text=part,
                )
            )
        return chunks

    def chunk_archive(
        self,
        *,
        members: list[tuple[str, str]],
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        """One archive member (e.g. `secedit.txt`) can become several
        chunks like any other text; `section` carries the member's own
        name so a citation says which config file a fact came from, not
        just the archive as a whole."""
        chunks: list[Chunk] = []
        for member_idx, (member_name, text) in enumerate(members):
            body = f"{member_name}\n\n{text}"
            for part_idx, part in enumerate(split_with_overlap(body, self.chunk_size, self.chunk_overlap)):
                chunks.append(
                    Chunk(
                        id=f"{document_id}::member{member_idx}::{part_idx}",
                        document_id=document_id,
                        document_title=document_title,
                        document_type=document_type,
                        version=version,
                        publication_date=publication_date,
                        page_start=0,
                        page_end=0,
                        section=member_name,
                        requirement_id=None,
                        testing_procedure_ids=(),
                        chunk_type="generic",
                        subsection=None,
                        parent_chunk_id=None,
                        source_file=source_file,
                        text=part,
                    )
                )
        return chunks

    def chunk_docx(
        self,
        *,
        sections: list[DocxSection],
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, section in enumerate(sections):
            if not section.text.strip():
                continue
            body = f"{section.heading}\n\n{section.text}" if section.heading else section.text
            for part_idx, part in enumerate(split_with_overlap(body, self.chunk_size, self.chunk_overlap)):
                chunks.append(
                    Chunk(
                        id=f"{document_id}::section{idx}::{part_idx}",
                        document_id=document_id,
                        document_title=document_title,
                        document_type=document_type,
                        version=version,
                        publication_date=publication_date,
                        page_start=0,
                        page_end=0,
                        section=section.heading,
                        requirement_id=None,
                        testing_procedure_ids=(),
                        chunk_type="generic",
                        subsection=None,
                        parent_chunk_id=None,
                        source_file=source_file,
                        text=part,
                    )
                )
        return chunks
