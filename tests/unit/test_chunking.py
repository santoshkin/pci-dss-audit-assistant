from app.chunking import GenericChunker, StructureAwareChunker, split_with_overlap
from app.ingestion.docx_parser import DocxSection
from app.ingestion.structure import GuidanceBlock, Requirement
from app.ingestion.structure import TestingProcedure as TP


def test_split_with_overlap_short_text_stays_one_part():
    assert split_with_overlap("short", chunk_size=100, chunk_overlap=10) == ["short"]


def test_split_with_overlap_respects_overlap_and_covers_whole_text():
    text = "a" * 250
    parts = split_with_overlap(text, chunk_size=100, chunk_overlap=20)
    assert len(parts) > 1
    # every character position must be covered by at least one part
    assert "".join(parts).count("a") >= len(text)
    # consecutive parts actually overlap
    assert parts[0][-20:] == parts[1][:20]


class TestStructureAwareChunker:
    def _requirement(self) -> Requirement:
        return Requirement(
            id="8.4.2",
            description="MFA is implemented for all access into the CDE.",
            customized_approach_objective="",
            applicability_notes="",
            page_start=203,
            page_end=203,
            section="8.4 Multi-factor authentication",
        )

    def test_requirement_and_testing_procedures_stay_in_one_chunk(self):
        req = self._requirement()
        tps = [
            TP(id="8.4.2.a", requirement_id="8.4.2", text="Examine configs.", page_start=203, page_end=203),
            TP(id="8.4.2.b", requirement_id="8.4.2", text="Interview staff.", page_start=203, page_end=203),
        ]
        chunker = StructureAwareChunker(chunk_size=2000, chunk_overlap=100)
        chunks = chunker.chunk(
            document_id="pci_dss_v4.0.1",
            document_title="PCI DSS",
            document_type="pci_dss",
            version="4.0.1",
            publication_date="2024",
            source_file="test.pdf",
            requirements=[req],
            testing_procedures=tps,
            guidance_blocks=[],
        )
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.requirement_id == "8.4.2"
        assert chunk.testing_procedure_ids == ("8.4.2.a", "8.4.2.b")
        assert chunk.chunk_type == "requirement"
        assert "MFA is implemented" in chunk.text
        assert "Examine configs." in chunk.text

    def test_guidance_becomes_child_chunk_linked_to_requirement(self):
        req = self._requirement()
        guidance = GuidanceBlock(
            requirement_id="8.4.2",
            sections={"Purpose": "Reduces the likelihood of a successful masquerading attack."},
            page_start=203,
            page_end=203,
        )
        chunker = StructureAwareChunker(chunk_size=2000, chunk_overlap=100)
        chunks = chunker.chunk(
            document_id="pci_dss_v4.0.1",
            document_title="PCI DSS",
            document_type="pci_dss",
            version="4.0.1",
            publication_date="2024",
            source_file="test.pdf",
            requirements=[req],
            testing_procedures=[],
            guidance_blocks=[guidance],
        )
        requirement_chunk = next(c for c in chunks if c.chunk_type == "requirement")
        guidance_chunk = next(c for c in chunks if c.chunk_type == "guidance")
        assert guidance_chunk.parent_chunk_id == requirement_chunk.id
        assert guidance_chunk.subsection == "Purpose"
        assert guidance_chunk.requirement_id == "8.4.2"

    def test_oversized_testing_procedures_split_into_sibling_chunks(self):
        req = self._requirement()
        tps = [
            TP(id=f"8.4.2.{c}", requirement_id="8.4.2", text="x" * 80, page_start=203, page_end=203)
            for c in "abcdefgh"
        ]
        chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(
            document_id="pci_dss_v4.0.1",
            document_title="PCI DSS",
            document_type="pci_dss",
            version="4.0.1",
            publication_date="2024",
            source_file="test.pdf",
            requirements=[req],
            testing_procedures=tps,
            guidance_blocks=[],
        )
        requirement_chunks = [c for c in chunks if c.chunk_type == "requirement"]
        assert len(requirement_chunks) > 1
        # every testing procedure ends up in exactly one sibling chunk
        all_tp_ids = [tp_id for c in requirement_chunks for tp_id in c.testing_procedure_ids]
        assert sorted(all_tp_ids) == sorted(tp.id for tp in tps)


class TestGenericChunker:
    def test_chunk_text_has_no_requirement_metadata(self):
        chunker = GenericChunker(chunk_size=1000, chunk_overlap=50)
        chunks = chunker.chunk_text(
            text="Интервью с администратором про MFA и ротацию паролей.",
            document_id="evidence_interview_transcript_x",
            document_title="interview",
            document_type="evidence",
            version="unknown",
            publication_date="unknown",
            source_file="interview.txt",
        )
        assert len(chunks) == 1
        assert chunks[0].requirement_id is None
        assert chunks[0].chunk_type == "generic"
        assert chunks[0].document_type == "evidence"

    def test_chunk_docx_uses_word_heading_styles_as_sections(self):
        sections = [
            DocxSection(heading=None, paragraphs=["Intro paragraph, no heading yet."]),
            DocxSection(heading="Password Policy", paragraphs=["Minimum length is 15 characters."]),
        ]
        chunker = GenericChunker(chunk_size=1000, chunk_overlap=50)
        chunks = chunker.chunk_docx(
            sections=sections,
            document_id="evidence_customer_document_x",
            document_title="ISMS Policy",
            document_type="evidence",
            version="unknown",
            publication_date="unknown",
            source_file="policy.docx",
        )
        assert len(chunks) == 2
        assert chunks[0].section is None
        assert chunks[1].section == "Password Policy"
        assert "Minimum length is 15 characters." in chunks[1].text

    def test_chunk_docx_skips_empty_sections(self):
        sections = [DocxSection(heading="Empty Heading", paragraphs=[])]
        chunker = GenericChunker(chunk_size=1000, chunk_overlap=50)
        chunks = chunker.chunk_docx(
            sections=sections,
            document_id="doc",
            document_title="doc",
            document_type="evidence",
            version="unknown",
            publication_date="unknown",
            source_file="doc.docx",
        )
        assert chunks == []
