from app.chat import format_context_entry, format_source, strip_model_sources_section


def test_format_source_requirement_chunk():
    payload = {
        "requirement_id": "8.4.2",
        "testing_procedure_ids": ("8.4.2.a",),
        "subsection": None,
        "version": "4.0.1",
        "page_start": 203,
        "page_end": 203,
    }
    assert format_source(payload) == "PCI DSS v4.0.1, Requirement 8.4.2, Testing Procedure 8.4.2.a, p. 203"


def test_format_source_requirement_chunk_unknown_version_and_page_range():
    payload = {
        "requirement_id": "8.4.2",
        "testing_procedure_ids": (),
        "subsection": "Purpose",
        "version": "unknown",
        "page_start": 203,
        "page_end": 204,
    }
    assert format_source(payload) == "PCI DSS, Requirement 8.4.2, Purpose, pp. 203-204"


def test_format_source_evidence_chunk_is_never_confused_with_standard():
    # PLAN.md section 2 isolation requirement: an evidence citation must be
    # unambiguously distinguishable from an official Standard/FAQ one -
    # this is what app/services/project_chat.py's merged answer relies on.
    payload = {
        "requirement_id": None,
        "document_type": "evidence",
        "document_title": "interview",
        "section": None,
        "version": "unknown",
        "page_start": 0,
        "page_end": 0,
    }
    result = format_source(payload)
    assert result == "Evidence: interview"
    assert "PCI DSS" not in result


def test_format_source_generic_faq_chunk():
    payload = {
        "requirement_id": None,
        "document_type": "faq",
        "document_title": "PCI DSS v4.x ROC Template FAQs",
        "section": "2.1 What is a ROC?",
        "version": "4.x",
        "page_start": 3,
        "page_end": 3,
    }
    assert format_source(payload) == "PCI DSS v4.x ROC Template FAQs, v4.x, 2.1 What is a ROC?, p. 3"


def test_format_context_entry_wraps_source_and_text():
    payload = {"requirement_id": None, "document_type": "evidence", "document_title": "x", "section": None,
               "version": "unknown", "page_start": 0, "page_end": 0, "text": "the actual chunk text"}
    entry = format_context_entry(payload)
    assert entry.startswith("[Evidence: x]\n")
    assert entry.endswith("the actual chunk text")


def test_strip_model_sources_section_removes_trailing_heading():
    generated = "Ответ по существу.\n\n## Источники\n\n* модель что-то придумала"
    assert strip_model_sources_section(generated) == "Ответ по существу."


def test_strip_model_sources_section_leaves_text_without_heading_untouched():
    generated = "Просто ответ без секции источников."
    assert strip_model_sources_section(generated) == generated
