"""Validates app/ingestion against the real PCI DSS v4.0.1 PDF.

Runs the full PdfLayoutParser -> StructureExtractor pipeline and reports:
- counts and any parser warnings (unresolved continuation markers, guidance
  anchor fallbacks),
- coverage: requirements with no testing procedure, testing procedures with
  no parent requirement, requirements with no guidance at all,
- the specific 1.2.3/1.2.4 page-break case identified during the smoke test
  (ARCHITECTURE.md section 8), to confirm Guidance no longer gets misattached
  to the wrong requirement across a page boundary,
- a manual-review sample.

Usage: .venv/bin/python scripts/ingestion_validation.py
"""

from pathlib import Path

from app.ingestion import PdfLayoutParser, StructureExtractor

PDF_PATH = Path("data/documents/pci_dss/PCI-DSS-v4_0_1.pdf")


def main() -> None:
    parser = PdfLayoutParser(PDF_PATH)
    metadata = parser.extract_metadata()
    print(f"Document: {metadata.title!r} version={metadata.version} date={metadata.date}")
    print()

    pages = list(parser.iter_pages())
    section_titles = parser.extract_section_titles()
    result = StructureExtractor().extract(pages, section_titles)

    print(f"Requirements: {len(result.requirements)}")
    print(f"Testing procedures: {len(result.testing_procedures)}")
    print(f"Guidance blocks: {len(result.guidance_blocks)}")
    print(f"Parser warnings: {len(result.warnings)}")
    for w in result.warnings[:20]:
        print(f"  - {w}")
    if len(result.warnings) > 20:
        print(f"  ... and {len(result.warnings) - 20} more")
    print()

    req_ids = {r.id for r in result.requirements}
    tp_orphans = [tp for tp in result.testing_procedures if tp.requirement_id not in req_ids]
    print(f"Testing procedures with no parent requirement: {len(tp_orphans)}")
    for tp in tp_orphans[:10]:
        print(f"  - {tp.id} (parent {tp.requirement_id!r} not found)")

    guided_ids = {g.requirement_id for g in result.guidance_blocks}
    unguided = [r for r in result.requirements if r.id not in guided_ids]
    print(f"Requirements with no guidance block at all: {len(unguided)} / {len(result.requirements)}")
    print()

    print("=== 1.2.3 / 1.2.4 page-break case (ARCHITECTURE.md section 8) ===")
    by_id = {r.id: r for r in result.requirements}
    guidance_by_id = {g.requirement_id: g for g in result.guidance_blocks}
    for rid in ("1.2.3", "1.2.4"):
        req = by_id.get(rid)
        gb = guidance_by_id.get(rid)
        print(f"-- Requirement {rid} --")
        if req:
            print(f"   pages {req.page_start}-{req.page_end}: {req.description[:90]}...")
        if gb:
            print(f"   guidance pages {gb.page_start}-{gb.page_end}, sections: {list(gb.sections)}")
            for heading, text in gb.sections.items():
                print(f"     [{heading}] {text[:100]}...")
        else:
            print("   (no guidance block found)")
    print()

    print("=== Sample of 10 requirements with full structure ===")
    sample_ids = ["1.1.1", "1.2.5", "3.2.1", "8.3.5", "11.4.1", "12.6.3.2"]
    tp_by_req: dict[str, list] = {}
    for tp in result.testing_procedures:
        tp_by_req.setdefault(tp.requirement_id, []).append(tp)

    for rid in sample_ids:
        req = by_id.get(rid)
        if not req:
            print(f"[{rid}] NOT FOUND")
            continue
        print(f"[{rid}] (p.{req.page_start}-{req.page_end}) {req.description}")
        if req.customized_approach_objective:
            print(f"   CAO: {req.customized_approach_objective[:120]}")
        if req.applicability_notes:
            print(f"   Applicability Notes: {req.applicability_notes[:120]}")
        for tp in tp_by_req.get(rid, []):
            print(f"   TP {tp.id}: {tp.text[:120]}")
        gb = guidance_by_id.get(rid)
        if gb:
            for heading, text in gb.sections.items():
                print(f"   Guidance/{heading}: {text[:120]}")
        print()


if __name__ == "__main__":
    main()
