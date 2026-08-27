"""Phase 0 smoke test (PLAN.md section 8, item 2).

Runs the candidate column-splitting approach over the real PCI DSS v4.0.1
PDF, extracts Requirement IDs and Testing Procedure IDs from the
Requirements/Testing Procedures/Guidance table pages, and prints counts plus
a sample for manual cross-checking against the source PDF before any
ingestion/chunking code is built on top of this approach.

Usage: .venv/bin/python scripts/pdf_structure_smoke_test.py
"""

import re
from pathlib import Path

import pdfplumber

PDF_PATH = Path("data/documents/pci_dss/PCI-DSS-v4_0_1.pdf")

# Column boundaries observed on the 3-column Requirements/Testing
# Procedures/Guidance pages (landscape, 792x612pt), see ARCHITECTURE.md.
COL1_END = 295  # Defined Approach Requirements
COL2_END = 505  # Defined Approach Testing Procedures
# remainder -> Guidance

TABLE_PAGE_MARKER = "Defined Approach Requirements"

REQ_ID_RE = re.compile(r"^(\d+\.\d+(?:\.\d+){0,2})\s+(.*)$")
TP_ID_RE = re.compile(r"^(\d+\.\d+(?:\.\d+){0,2}\.[a-z])\s+(.*)$")


def is_table_page(text: str) -> bool:
    return TABLE_PAGE_MARKER in text


def extract_ids(column_text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    found = []
    # requirement/testing-procedure IDs start a new logical entry; text may
    # wrap onto following lines without a leading ID, so we only match lines
    # that start with an ID and keep the first line of body text.
    for line in column_text.splitlines():
        line = line.strip()
        m = pattern.match(line)
        if m:
            found.append((m.group(1), m.group(2)))
    return found


def main() -> None:
    requirements: list[tuple[int, str, str]] = []
    testing_procedures: list[tuple[int, str, str]] = []
    table_pages = 0

    with pdfplumber.open(PDF_PATH) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not is_table_page(text):
                continue
            table_pages += 1

            col1 = page.crop((0, 0, COL1_END, page.height)).extract_text() or ""
            col2 = page.crop((COL1_END, 0, COL2_END, page.height)).extract_text() or ""

            for req_id, body in extract_ids(col1, REQ_ID_RE):
                requirements.append((page_num, req_id, body))
            for tp_id, body in extract_ids(col2, TP_ID_RE):
                testing_procedures.append((page_num, tp_id, body))

    print(f"Table pages detected: {table_pages}")
    print(f"Requirement IDs extracted (with duplicates across pages): {len(requirements)}")
    print(f"Distinct Requirement IDs: {len({r[1] for r in requirements})}")
    print(f"Testing Procedure IDs extracted: {len(testing_procedures)}")
    print(f"Distinct Testing Procedure IDs: {len({t[1] for t in testing_procedures})}")
    print()

    print("=== Sample of 15 Requirement IDs for manual cross-check ===")
    sample_idx = [0, 5, 10, 20, 40, 80, 120, 160, 200, 240, 260, 280, 300, 310, len(requirements) - 1]
    seen = set()
    shown = 0
    for i in sample_idx:
        if i < 0 or i >= len(requirements):
            continue
        page_num, req_id, body = requirements[i]
        if req_id in seen:
            continue
        seen.add(req_id)
        print(f"[p.{page_num}] {req_id}: {body}")
        shown += 1
    print(f"({shown} shown)")

    print()
    print("=== First 20 Requirement IDs in document order (sanity check) ===")
    for page_num, req_id, body in requirements[:20]:
        print(f"[p.{page_num}] {req_id}: {body}")


if __name__ == "__main__":
    main()
