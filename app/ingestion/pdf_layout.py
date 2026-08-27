"""Page-level layout parsing for the PCI DSS "Requirements and Testing
Procedures" PDF (v4.0.1 and, expected to stay stable, later minor
revisions of the same template).

Coordinates below were derived and validated against the real document —
see ARCHITECTURE.md section 8 for the smoke-test that established them.
The 3-column table layout (Requirements / Testing Procedures / Guidance)
is specific to this PCI DSS document template; it is not a general-purpose
PDF parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pdfplumber

# The per-row "Defined Approach Requirements" header is absent on pages
# that are pure Guidance continuation with no new row starting anywhere on
# them (~20 pages in the real document, e.g. one made entirely of
# "1.2.4 (continued) ..." guidance prose) - the page-top banner is present
# on every content page of this template, including those, and absent from
# genuine narrative pages (chapter intros, "Sections" overviews, appendix
# overviews), so it is the reliable table-page signal, not the header.
TABLE_PAGE_MARKER = "Requirements and Testing Procedures Guidance"

# Default column boundaries in PDF points, landscape page (792x612), used
# only as a fallback when a page's own header tokens can't be located (see
# `_detect_column_boundaries`). Column start actually varies by a dozen-odd
# points between the main body and Appendix A pages (observed: column 2
# starts at x0=298 on body pages vs. x0=285 on Appendix A pages) - a fixed
# split clips characters off whichever column's text sits closest to it, so
# boundaries are detected per page from the "Defined ... Testing
# Procedures" / "Purpose" header tokens instead of hardcoded here.
DEFAULT_COL1_END = 295.0
DEFAULT_COL2_END = 505.0
# Second header cluster ("Defined Approach Testing Procedures") always sits
# well right of the first ("Defined Approach Requirements") on the same
# line; anything above this x0 threshold is assumed to be the second.
COL2_HEADER_MIN_X0 = 200.0
# Safety margin subtracted from a detected header's x0 so the crop boundary
# falls just before that column's own left text margin, not on top of it.
BOUNDARY_MARGIN = 3.0

# Content band in y, excludes the repeating page banner (top ~86-97) and
# the footer (top ~545-568, consistent across all 261 table pages checked).
CONTENT_TOP = 100.0
CONTENT_BOTTOM = 540.0

# Anchor tokens are only trusted as row-start markers when they sit at the
# column's real left text margin (observed ~78pt from the page edge, not
# from the crop box, which is why this is an absolute threshold) - body
# text that happens to mention a bare requirement number rarely starts a
# line flush left.
COL1_LEFT_MARGIN_MAX_X0 = 90.0

# Appendix A (A1-A3, "Additional PCI DSS Requirements for ...") uses a
# letter-prefixed numbering scheme (e.g. "A1.1.1", "A3.2.3") instead of the
# main body's plain numeric one - both are matched by the optional leading
# uppercase letter here.
REQ_ID_TOKEN_RE = re.compile(r"^[A-Z]?\d+\.\d+(?:\.\d+){0,3}$")
GUIDANCE_HEADINGS = ("Purpose", "Good Practice", "Examples", "Definitions", "Further Information")

_VERSION_RE = re.compile(r"Requirements and Testing Procedures,\s*(v[\d.]+)")
_DATE_RE = re.compile(r"(v[\d.]+)\s+([A-Za-z]+ \d{4})")

# Each principal requirement/appendix opens with a single-column overview
# page listing its 2-level sections in full (e.g. "1.1 Processes and
# mechanisms for installing and maintaining network security controls are
# defined and understood."). The table pages themselves only ever show a
# same-text banner truncated to one line, so this is the only place the
# full section title is available.
SECTION_LIST_HEADER = "Sections"
SECTION_LINE_RE = re.compile(r"^([A-Z]?\d+\.\d+)\s+(.+)$")


@dataclass(frozen=True)
class WordAnchor:
    text: str
    top: float


@dataclass(frozen=True)
class PageColumns:
    page_number: int
    is_table_page: bool
    requirements_text: str
    testing_procedures_text: str
    guidance_text: str
    # Anchor y-positions for row-start tokens, used by StructureExtractor to
    # correlate a Guidance "Purpose" block with the requirement row it
    # belongs to when the two columns have drifted apart across a page
    # break (see ARCHITECTURE.md section 8, "Открытый вопрос").
    requirement_anchors: tuple[WordAnchor, ...]
    guidance_heading_anchors: tuple[WordAnchor, ...]


@dataclass(frozen=True)
class DocumentMetadata:
    title: str
    version: str
    date: str


class PdfLayoutParser:
    """Splits each page of the PCI DSS PDF into its three logical columns."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def iter_pages(self) -> Iterator[PageColumns]:
        col1_end, col2_end = DEFAULT_COL1_END, DEFAULT_COL2_END
        with pdfplumber.open(self.path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                full_text = page.extract_text() or ""
                is_table_page = TABLE_PAGE_MARKER in full_text
                if not is_table_page:
                    yield PageColumns(
                        page_number=page_number,
                        is_table_page=False,
                        requirements_text=full_text,
                        testing_procedures_text="",
                        guidance_text="",
                        requirement_anchors=(),
                        guidance_heading_anchors=(),
                    )
                    continue

                col1_end, col2_end = self._detect_column_boundaries(page, fallback=(col1_end, col2_end))

                col1 = page.crop((0, CONTENT_TOP, col1_end, CONTENT_BOTTOM))
                col2 = page.crop((col1_end, CONTENT_TOP, col2_end, CONTENT_BOTTOM))
                col3 = page.crop((col2_end, CONTENT_TOP, page.width, CONTENT_BOTTOM))

                yield PageColumns(
                    page_number=page_number,
                    is_table_page=True,
                    requirements_text=col1.extract_text() or "",
                    testing_procedures_text=col2.extract_text() or "",
                    guidance_text=col3.extract_text() or "",
                    requirement_anchors=self._requirement_anchors(col1),
                    guidance_heading_anchors=self._guidance_heading_anchors(col3),
                )

    @staticmethod
    def _detect_column_boundaries(page, fallback: tuple[float, float]) -> tuple[float, float]:
        """Locates this page's actual column split points from the
        "Defined Approach Testing Procedures" / "Purpose" header tokens,
        instead of trusting a fixed pixel constant - see the module-level
        comment on DEFAULT_COL1_END for why a fixed split clips characters
        on some pages (e.g. Appendix A)."""
        words = page.extract_words()
        col2_header_x0 = min(
            (w["x0"] for w in words if w["text"] == "Defined" and w["x0"] >= COL2_HEADER_MIN_X0),
            default=None,
        )
        col3_header_x0 = min((w["x0"] for w in words if w["text"] == "Purpose"), default=None)

        col1_end = col2_header_x0 - BOUNDARY_MARGIN if col2_header_x0 is not None else fallback[0]
        col2_end = col3_header_x0 - BOUNDARY_MARGIN if col3_header_x0 is not None else fallback[1]
        return col1_end, col2_end

    def extract_metadata(self) -> DocumentMetadata:
        with pdfplumber.open(self.path) as pdf:
            title = (pdf.pages[0].extract_text() or "").splitlines()[0].strip()
            for page in pdf.pages:
                text = page.extract_text() or ""
                if TABLE_PAGE_MARKER not in text:
                    continue
                version_match = _VERSION_RE.search(text)
                date_match = _DATE_RE.search(text)
                if version_match and date_match:
                    return DocumentMetadata(
                        title=title,
                        version=version_match.group(1).lstrip("v"),
                        date=date_match.group(2),
                    )
        raise ValueError(f"Could not extract version/date footer from {self.path}")

    def extract_section_titles(self) -> dict[str, str]:
        """Full-text lookup for 2-level section IDs (e.g. "1.1" ->
        "Processes and mechanisms for installing and maintaining network
        security controls are defined and understood."), read from each
        principal requirement/appendix's overview page. See
        `SECTION_LIST_HEADER` for why the table pages themselves can't be
        used for this."""
        titles: dict[str, str] = {}
        with pdfplumber.open(self.path) as pdf:
            for page in pdf.pages:
                lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
                if SECTION_LIST_HEADER not in lines:
                    continue
                start = lines.index(SECTION_LIST_HEADER) + 1
                for line in lines[start:]:
                    match = SECTION_LINE_RE.match(line)
                    if not match:
                        break
                    titles[match.group(1)] = f"{match.group(1)} {match.group(2)}"
        return titles

    @staticmethod
    def _requirement_anchors(col1_crop) -> tuple[WordAnchor, ...]:
        anchors = []
        for word in col1_crop.extract_words():
            if word["x0"] > COL1_LEFT_MARGIN_MAX_X0:
                continue
            if REQ_ID_TOKEN_RE.fullmatch(word["text"]):
                anchors.append(WordAnchor(text=word["text"], top=word["top"]))
        return tuple(anchors)

    @staticmethod
    def _guidance_heading_anchors(col3_crop) -> tuple[WordAnchor, ...]:
        # Headings are 1-3 words (e.g. "Further Information"); group words
        # into lines by clustering on `top`, then match the full line text.
        words = col3_crop.extract_words()
        lines: dict[int, list[dict]] = {}
        for word in words:
            key = round(word["top"])
            lines.setdefault(key, []).append(word)

        anchors = []
        for top_key, line_words in lines.items():
            line_words.sort(key=lambda w: w["x0"])
            line_text = " ".join(w["text"] for w in line_words)
            if line_text in GUIDANCE_HEADINGS:
                anchors.append(WordAnchor(text=line_text, top=float(top_key)))
        anchors.sort(key=lambda a: a.top)
        return tuple(anchors)
