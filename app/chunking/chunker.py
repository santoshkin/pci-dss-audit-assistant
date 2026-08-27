"""Structure-aware chunking (PLAN.md section 7).

Chunk boundaries follow the document's own structure instead of a fixed
character count: a Requirement is always kept together with its Testing
Procedures in one chunk (splitting them apart risks retrieving a
requirement without knowing how it's tested, or vice versa - PLAN.md
explicitly calls this out), while each Guidance heading (Purpose, Good
Practice, Examples, Definitions, Further Information) becomes its own
child chunk linked back to the requirement's chunk via `parent_chunk_id`,
since guidance is supplementary explanation that benefits from being
retrievable independently of the requirement text itself.

`chunk_size`/`chunk_overlap` are used only as a last-resort safety net for
the rare oversized requirement (packing testing procedures into sibling
chunks once the target size is exceeded) or oversized guidance section
(sliding-window split) - never as the primary splitting strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.pdf_layout import GUIDANCE_HEADINGS
from app.ingestion.structure import GuidanceBlock, Requirement, TestingProcedure

# A single oversized chunk still embeds and retrieves fine; this is only a
# guard against a pathological guidance section far beyond anything seen in
# the real PCI DSS v4.0.1 corpus (max observed: ~2.2k chars, see
# ARCHITECTURE.md) blowing up embedding quality.
MAX_OVERSIZE_FACTOR = 3


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    document_type: str
    version: str
    publication_date: str
    page_start: int
    page_end: int
    section: str | None
    requirement_id: str | None
    testing_procedure_ids: tuple[str, ...]
    chunk_type: str  # "requirement" | "guidance" | "definition"
    subsection: str | None  # guidance heading, e.g. "Purpose"
    parent_chunk_id: str | None
    source_file: str
    text: str


def split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return parts


class StructureAwareChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        *,
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
        requirements: list[Requirement],
        testing_procedures: list[TestingProcedure],
        guidance_blocks: list[GuidanceBlock],
    ) -> list[Chunk]:
        tp_by_requirement: dict[str, list[TestingProcedure]] = {}
        for tp in testing_procedures:
            tp_by_requirement.setdefault(tp.requirement_id, []).append(tp)
        guidance_by_requirement = {g.requirement_id: g for g in guidance_blocks}

        chunks: list[Chunk] = []
        for req in requirements:
            req_chunks = self._chunk_requirement(
                req, tp_by_requirement.get(req.id, []), document_id, document_title,
                document_type, version, publication_date, source_file,
            )
            chunks.extend(req_chunks)
            parent_id = req_chunks[0].id

            guidance = guidance_by_requirement.get(req.id)
            if guidance is None:
                continue
            chunks.extend(
                self._chunk_guidance(
                    req, guidance, parent_id, document_id, document_title,
                    document_type, version, publication_date, source_file,
                )
            )
        return chunks

    def _chunk_requirement(
        self,
        req: Requirement,
        tps: list[TestingProcedure],
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        header_parts = [f"Requirement {req.id}: {req.description}"]
        if req.customized_approach_objective:
            header_parts.append(f"Customized Approach Objective: {req.customized_approach_objective}")
        if req.applicability_notes:
            header_parts.append(f"Applicability Notes: {req.applicability_notes}")
        header = "\n\n".join(header_parts)

        tps_sorted = sorted(tps, key=lambda tp: tp.id)

        groups: list[list[TestingProcedure]] = [[]]
        current_len = len(header)
        for tp in tps_sorted:
            tp_text_len = len(f"\n\nTesting Procedure {tp.id}: {tp.text}")
            if groups[-1] and current_len + tp_text_len > self.chunk_size:
                groups.append([])
                current_len = len(header)
            groups[-1].append(tp)
            current_len += tp_text_len

        page_start = min([req.page_start] + [tp.page_start for tp in tps])
        page_end = max([req.page_end] + [tp.page_end for tp in tps])

        chunks = []
        for idx, group in enumerate(groups):
            parts = [header]
            parts.extend(f"Testing Procedure {tp.id}: {tp.text}" for tp in group)
            suffix = "" if len(groups) == 1 else f"#{idx}"
            chunks.append(
                Chunk(
                    id=f"{document_id}::{req.id}::requirement{suffix}",
                    document_id=document_id,
                    document_title=document_title,
                    document_type=document_type,
                    version=version,
                    publication_date=publication_date,
                    page_start=page_start,
                    page_end=page_end,
                    section=req.section,
                    requirement_id=req.id,
                    testing_procedure_ids=tuple(tp.id for tp in group),
                    chunk_type="requirement",
                    subsection=None,
                    parent_chunk_id=None,
                    source_file=source_file,
                    text="\n\n".join(parts),
                )
            )
        return chunks

    def _chunk_guidance(
        self,
        req: Requirement,
        guidance: GuidanceBlock,
        parent_chunk_id: str,
        document_id: str,
        document_title: str,
        document_type: str,
        version: str,
        publication_date: str,
        source_file: str,
    ) -> list[Chunk]:
        chunks = []
        for heading in GUIDANCE_HEADINGS:
            text = guidance.sections.get(heading)
            if not text:
                continue
            chunk_type = "definition" if heading == "Definitions" else "guidance"
            body = f"PCI DSS Requirement {req.id} — {heading}\n\n{text}"

            parts = split_with_overlap(body, self.chunk_size * MAX_OVERSIZE_FACTOR, self.chunk_overlap)
            for idx, part in enumerate(parts):
                suffix = "" if len(parts) == 1 else f"#{idx}"
                chunks.append(
                    Chunk(
                        id=f"{document_id}::{req.id}::{chunk_type}::{heading}{suffix}",
                        document_id=document_id,
                        document_title=document_title,
                        document_type=document_type,
                        version=version,
                        publication_date=publication_date,
                        page_start=guidance.page_start,
                        page_end=guidance.page_end,
                        section=req.section,
                        requirement_id=req.id,
                        testing_procedure_ids=(),
                        chunk_type=chunk_type,
                        subsection=heading,
                        parent_chunk_id=parent_chunk_id,
                        source_file=source_file,
                        text=part,
                    )
                )
        return chunks
