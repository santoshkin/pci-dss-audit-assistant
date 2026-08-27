"""Validates app/chunking on top of the real PCI DSS v4.0.1 extraction.

Usage: .venv/bin/python scripts/chunking_validation.py
"""

from pathlib import Path

from app.chunking import StructureAwareChunker
from app.config import get_settings
from app.ingestion import PdfLayoutParser, StructureExtractor

PDF_PATH = Path("data/documents/pci_dss/PCI-DSS-v4_0_1.pdf")


def main() -> None:
    settings = get_settings()
    parser = PdfLayoutParser(PDF_PATH)
    metadata = parser.extract_metadata()
    pages = list(parser.iter_pages())
    section_titles = parser.extract_section_titles()
    extraction = StructureExtractor().extract(pages, section_titles)

    chunker = StructureAwareChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk(
        document_id=f"pci_dss_v{metadata.version}",
        document_title=metadata.title,
        document_type="pci_dss",
        version=metadata.version,
        publication_date=metadata.date,
        source_file=str(PDF_PATH),
        requirements=extraction.requirements,
        testing_procedures=extraction.testing_procedures,
        guidance_blocks=extraction.guidance_blocks,
    )

    print(f"Total chunks: {len(chunks)}")
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print("By type:", by_type)

    lengths = sorted((len(c.text) for c in chunks), reverse=True)
    print(f"Length: max={lengths[0]} p95={lengths[len(lengths)//20]} median={lengths[len(lengths)//2]}")

    multi_part_reqs = [c for c in chunks if c.chunk_type == "requirement" and "#" in c.id]
    print(f"Requirement chunks split into siblings: {len(multi_part_reqs)}")

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids!"

    req_chunks = [c for c in chunks if c.requirement_id == "1.2.5"]
    print()
    print("=== All chunks for Requirement 1.2.5 ===")
    for c in req_chunks:
        print(f"[{c.id}] type={c.chunk_type} subsection={c.subsection} parent={c.parent_chunk_id} "
              f"pages={c.page_start}-{c.page_end} section={c.section!r}")
        print(f"   {c.text[:200]}")
        print()

    print("=== Largest requirement chunk (checking TP-boundary split, not mid-sentence) ===")
    largest_req = max((c for c in chunks if c.chunk_type == "requirement"), key=lambda c: len(c.text))
    print(f"[{largest_req.id}] len={len(largest_req.text)} TPs={largest_req.testing_procedure_ids}")
    print(largest_req.text[:400])

    split_reqs = {c.requirement_id for c in chunks if c.chunk_type == "requirement" and "#" in c.id}
    if split_reqs:
        sample_id = sorted(split_reqs)[0]
        siblings = [c for c in chunks if c.requirement_id == sample_id and c.chunk_type == "requirement"]
        print()
        print(f"=== Split requirement example: {sample_id} ({len(siblings)} sibling chunks) ===")
        for c in siblings:
            print(f"[{c.id}] TPs={c.testing_procedure_ids} len={len(c.text)}")


if __name__ == "__main__":
    main()
