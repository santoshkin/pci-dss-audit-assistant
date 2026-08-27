"""Structure extraction for the PCI DSS Requirements/Testing
Procedures/Guidance table pages.

The tricky part (see ARCHITECTURE.md section 8) is that a single logical
table row can span multiple PDF pages, and the three columns do not
necessarily finish that row on the same page: Guidance is usually longer
than Requirements+Testing Procedures, so its content for row N is still
being emitted on a page where column 1 has already moved on to row N+1.
This module resolves that by anchoring each Guidance "Purpose" section to
the requirement whose row-start token sits at or just above it on the same
page (`requirement_anchors`/`guidance_heading_anchors` from
`PdfLayoutParser`), and by carrying accumulation state across page
boundaries for the plain-text continuations in between headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.pdf_layout import GUIDANCE_HEADINGS, PageColumns, WordAnchor

# Optional leading uppercase letter covers Appendix A's "A1.1.1"-style IDs
# alongside the main body's plain numeric ones (see pdf_layout.REQ_ID_TOKEN_RE).
# Up to 5 numeric segments total to cover the deepest observed ID
# (Requirement 9.5.1.2.1). The optional trailing "\.?" absorbs a stray
# period some rows print right after the ID (e.g. "9.4.1. Examine...").
REQ_LINE_RE = re.compile(r"^([A-Z]?\d+\.\d+(?:\.\d+){0,3})\.?\s+(.*)$")
# The trailing ".a"/".b" suffix is only present when a requirement has more
# than one testing procedure; a requirement with exactly one reuses its own
# bare ID for the procedure (e.g. "1.1.1 Examine documentation...").
TP_LINE_RE = re.compile(r"^([A-Z]?\d+\.\d+(?:\.\d+){0,3}(?:\.[a-z])?)\.?\s+(.*)$")
TP_SUFFIX_RE = re.compile(r"\.[a-z]$")

REQ_HEADER = "Defined Approach Requirements"
TP_HEADER = "Defined Approach Testing Procedures"
CUSTOMIZED_APPROACH_HEADER = "Customized Approach Objective"
APPLICABILITY_NOTES_HEADER = "Applicability Notes"

CONTINUATION_MARKER = "(continued"
CONTINUATION_LINE = "(continued on next page)"
# Pages that are pure Guidance continuation (no new row starts anywhere on
# them) sometimes open with an explicit "<id> (continued) <text>" marker
# naming the requirement being continued - a more reliable signal than the
# anchor-distance fallback in _resolve_guidance_owner when present.
GUIDANCE_CONTINUATION_RE = re.compile(r"^([A-Z]?\d+\.\d+(?:\.\d+){0,3})\s*\(continued\)\s*(.*)$")
SECTION_PREFIX_RE = re.compile(r"^[A-Z]?\d+\.\d+")

# A row's "Defined Approach Requirements" header line pushes column 1's ID
# token down relative to column 3's "Purpose" heading for the same row (no
# equivalent header repeats in column 3) - observed offset ~20-22pt, so we
# match by nearest absolute distance rather than requiring the anchor to be
# strictly above the heading. Beyond this distance we no longer trust any
# on-page anchor and treat the heading as continuing the still-open block
# from a previous page instead (points).
MAX_ANCHOR_DISTANCE = 50.0


@dataclass
class Requirement:
    id: str
    description: str
    customized_approach_objective: str
    applicability_notes: str
    page_start: int
    page_end: int
    # The decorative one-line section restatement that precedes this row's
    # own header (e.g. "3.2 Storage of account data is kept to a minimum.")
    # - None if this requirement is the first thing on its page, so no such
    # banner was seen yet.
    section: str | None = None


@dataclass
class TestingProcedure:
    id: str
    requirement_id: str
    text: str
    page_start: int
    page_end: int


@dataclass
class GuidanceBlock:
    requirement_id: str
    sections: dict[str, str] = field(default_factory=dict)
    page_start: int = 0
    page_end: int = 0


@dataclass
class ExtractionResult:
    requirements: list[Requirement]
    testing_procedures: list[TestingProcedure]
    guidance_blocks: list[GuidanceBlock]
    warnings: list[str]


def _append_text(existing: str, addition: str) -> str:
    return f"{existing} {addition}" if existing else addition


def _resolve_guidance_owner(
    heading_top: float | None,
    requirement_anchors: tuple[WordAnchor, ...],
    known_requirement_ids: dict,
    fallback_req_id: str | None,
    warnings: list[str],
    page_number: int,
) -> str | None:
    # Anchors include every bare-number token at the column's left margin,
    # which also catches the decorative 2-level section banner (e.g. "6.4
    # Public-facing web applications..."). That banner is never a real
    # requirement, but it can sit visually closer to a "Purpose" heading
    # than the real row it belongs to (e.g. "6.4" at top=112 vs. the real
    # "6.4.1" at top=156, with "Purpose" at top=133) - restricting
    # candidates to IDs already confirmed as real requirements avoids
    # misattributing guidance to a banner and leaving the real row empty.
    requirement_anchors = tuple(a for a in requirement_anchors if a.text in known_requirement_ids)
    if heading_top is not None and requirement_anchors:
        nearest = min(requirement_anchors, key=lambda a: abs(a.top - heading_top))
        if abs(nearest.top - heading_top) <= MAX_ANCHOR_DISTANCE:
            return nearest.text

    if fallback_req_id is None:
        warnings.append(
            f"p.{page_number}: 'Purpose' heading has no nearby requirement anchor and no "
            f"open requirement to fall back to; guidance block dropped"
        )
        return None

    warnings.append(
        f"p.{page_number}: 'Purpose' at top={heading_top}: no requirement anchor within "
        f"{MAX_ANCHOR_DISTANCE}pt; treated as continuation of still-open requirement "
        f"{fallback_req_id!r}"
    )
    return fallback_req_id


class StructureExtractor:
    """Consumes `PageColumns` in page order and builds the structured
    Requirement / TestingProcedure / GuidanceBlock records."""

    def extract(self, pages: list[PageColumns], section_titles: dict[str, str] | None = None) -> ExtractionResult:
        requirements: dict[str, Requirement] = {}
        requirements_order: list[str] = []
        testing_procedures: dict[str, TestingProcedure] = {}
        guidance: dict[str, GuidanceBlock] = {}
        warnings: list[str] = []

        current_req_id: str | None = None
        current_req_field: str | None = None
        req_awaiting_id = False
        current_section: str | None = None
        current_tp_id: str | None = None
        current_guidance_req_id: str | None = None
        current_guidance_heading: str | None = None

        for page in pages:
            if not page.is_table_page:
                continue

            current_req_id, current_req_field, req_awaiting_id, current_section = self._parse_requirements_column(
                page, requirements, requirements_order, current_req_id, current_req_field, req_awaiting_id,
                current_section, section_titles or {}, warnings,
            )
            current_tp_id = self._parse_testing_procedures_column(
                page, testing_procedures, current_tp_id, warnings
            )
            current_guidance_req_id, current_guidance_heading = self._parse_guidance_column(
                page, guidance, requirements, current_guidance_req_id, current_guidance_heading, warnings
            )

        return ExtractionResult(
            requirements=list(requirements.values()),
            testing_procedures=list(testing_procedures.values()),
            guidance_blocks=list(guidance.values()),
            warnings=warnings,
        )

    @staticmethod
    def _parse_requirements_column(
        page: PageColumns,
        requirements: dict[str, Requirement],
        requirements_order: list[str],
        current_req_id: str | None,
        current_req_field: str | None,
        awaiting_id: bool,
        current_section: str | None,
        section_titles: dict[str, str],
        warnings: list[str],
    ) -> tuple[str | None, str | None, bool, str | None]:
        # `awaiting_id` means the previous line was a "Defined Approach
        # Requirements" header, so this line is expected to be a new
        # requirement's ID (or a "(continued)" marker for one already seen).
        # Without that gate, a plain cross-reference in running text (e.g.
        # "...will replace Requirement\n6.4.1 once its effective date...")
        # would be misread as a brand new requirement, silently clobbering
        # the real one - see ARCHITECTURE.md section 8 for the case that
        # surfaced this (Requirement 6.4.1's own text was overwritten by a
        # 6.4.1 cross-reference inside 6.4.2's Applicability Notes).
        lines = [ln.strip() for ln in page.requirements_text.splitlines() if ln.strip()]

        for idx, line in enumerate(lines):
            if line == REQ_HEADER:
                awaiting_id = True
                continue
            if line == CUSTOMIZED_APPROACH_HEADER:
                current_req_field = "customized_approach_objective"
                continue
            if line == APPLICABILITY_NOTES_HEADER:
                current_req_field = "applicability_notes"
                continue

            match = REQ_LINE_RE.match(line)
            if match:
                req_id, body = match.group(1), match.group(2).strip()
                is_continuation_marker = body.lower().startswith(CONTINUATION_MARKER)
                next_is_header = idx + 1 < len(lines) and lines[idx + 1] == REQ_HEADER

                if not awaiting_id and not is_continuation_marker and next_is_header:
                    # Decorative running restatement of the parent
                    # objective, repeated at the top of a page within a
                    # requirement group (e.g. "3.2 Storage of account data
                    # is kept to a minimum."); always immediately followed
                    # by the row header, never structured content itself -
                    # but it's the closest thing to a section title this
                    # template has, so keep it for the next requirement(s).
                    current_section = f"{req_id} {body}"
                    continue

                if is_continuation_marker:
                    if req_id in requirements:
                        requirements[req_id].page_end = page.page_number
                    else:
                        warnings.append(
                            f"p.{page.page_number}: '(continued)' marker for unseen requirement {req_id!r}"
                        )
                    current_req_id, current_req_field = req_id, None
                    awaiting_id = False
                    continue

                if awaiting_id:
                    current_req_id, current_req_field = req_id, "description"
                    prefix_match = SECTION_PREFIX_RE.match(req_id)
                    section = (
                        section_titles.get(prefix_match.group(0)) if prefix_match else None
                    ) or current_section
                    requirements[req_id] = Requirement(
                        id=req_id,
                        description=body,
                        customized_approach_objective="",
                        applicability_notes="",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section=section,
                    )
                    requirements_order.append(req_id)
                    awaiting_id = False
                    continue

                # A bare requirement-number-shaped token that isn't a new
                # entry (not awaiting one) and isn't the decorative banner:
                # a cross-reference to another requirement that happened to
                # start a wrapped line. Falls through to plain continuation
                # text below.

            if awaiting_id:
                warnings.append(
                    f"p.{page.page_number}: expected a requirement ID after '{REQ_HEADER}', found {line!r}"
                )
                awaiting_id = False

            if current_req_id is None or current_req_field is None:
                continue
            req = requirements[current_req_id]
            req.page_end = page.page_number
            setattr(req, current_req_field, _append_text(getattr(req, current_req_field), line))

        return current_req_id, current_req_field, awaiting_id, current_section

    @staticmethod
    def _parse_testing_procedures_column(
        page: PageColumns,
        testing_procedures: dict[str, TestingProcedure],
        current_tp_id: str | None,
        warnings: list[str],
    ) -> str | None:
        # Unlike column 1, "Defined Approach Testing Procedures" is printed
        # once per row, not once per procedure - a row with three
        # procedures (e.g. 3.2.1.a/.b/.c) lists all three directly under a
        # single header. So, unlike _parse_requirements_column, a new entry
        # is recognized from any TP_LINE_RE match, not gated behind having
        # just seen the header - that gate was tried and caused every
        # procedure after the first in a row to be silently merged into it.
        lines = [ln.strip() for ln in page.testing_procedures_text.splitlines() if ln.strip()]

        for line in lines:
            if line == TP_HEADER:
                continue

            match = TP_LINE_RE.match(line)
            if match:
                tp_id, body = match.group(1), match.group(2).strip()
                is_continuation_marker = body.lower().startswith(CONTINUATION_MARKER)

                if is_continuation_marker:
                    if tp_id in testing_procedures:
                        testing_procedures[tp_id].page_end = page.page_number
                    else:
                        warnings.append(
                            f"p.{page.page_number}: '(continued)' marker for unseen testing procedure {tp_id!r}"
                        )
                    current_tp_id = tp_id
                    continue

                current_tp_id = tp_id
                testing_procedures[tp_id] = TestingProcedure(
                    id=tp_id,
                    requirement_id=TP_SUFFIX_RE.sub("", tp_id),
                    text=body,
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
                continue

            if current_tp_id is None:
                continue
            tp = testing_procedures[current_tp_id]
            tp.page_end = page.page_number
            tp.text = _append_text(tp.text, line)

        return current_tp_id

    @staticmethod
    def _parse_guidance_column(
        page: PageColumns,
        guidance: dict[str, GuidanceBlock],
        requirements: dict[str, Requirement],
        current_guidance_req_id: str | None,
        current_guidance_heading: str | None,
        warnings: list[str],
    ) -> tuple[str | None, str | None]:
        anchor_idx = 0
        anchors = page.guidance_heading_anchors

        for raw_line in page.guidance_text.splitlines():
            line = raw_line.strip()
            if not line or line == CONTINUATION_LINE:
                continue

            continuation_match = GUIDANCE_CONTINUATION_RE.match(line)
            if continuation_match:
                marker_id, remainder = continuation_match.group(1), continuation_match.group(2).strip()
                if marker_id in requirements:
                    current_guidance_req_id = marker_id
                    if marker_id not in guidance:
                        guidance[marker_id] = GuidanceBlock(
                            requirement_id=marker_id, page_start=page.page_number, page_end=page.page_number
                        )
                else:
                    warnings.append(
                        f"p.{page.page_number}: guidance '(continued)' marker for unseen requirement {marker_id!r}"
                    )
                line = remainder
                if not line:
                    continue

            if line in GUIDANCE_HEADINGS:
                heading_top = anchors[anchor_idx].top if anchor_idx < len(anchors) else None
                anchor_idx += 1

                if line == "Purpose":
                    owner = _resolve_guidance_owner(
                        heading_top, page.requirement_anchors, requirements, current_guidance_req_id,
                        warnings, page.page_number,
                    )
                    current_guidance_req_id = owner
                    if owner is not None and owner not in guidance:
                        guidance[owner] = GuidanceBlock(
                            requirement_id=owner, page_start=page.page_number, page_end=page.page_number
                        )

                current_guidance_heading = line
                if current_guidance_req_id is not None:
                    block = guidance[current_guidance_req_id]
                    block.sections.setdefault(current_guidance_heading, "")
                    block.page_end = page.page_number
                continue

            if current_guidance_req_id is None or current_guidance_heading is None:
                continue

            block = guidance[current_guidance_req_id]
            block.page_end = page.page_number
            block.sections[current_guidance_heading] = _append_text(
                block.sections.get(current_guidance_heading, ""), line
            )

        return current_guidance_req_id, current_guidance_heading
