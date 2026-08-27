from app.retrieval.requirement_lookup import extract_requirement_id_candidates, requirement_family_prefix


def test_extracts_three_and_four_segment_ids():
    assert extract_requirement_id_candidates("Что говорит 8.4.2 про MFA?") == ["8.4.2"]
    assert extract_requirement_id_candidates("Требование 9.5.1.2.1 сложное") == ["9.5.1.2.1"]


def test_ignores_version_numbers_and_bare_sections():
    # "4.0.1" is a version number, "8.4" a two-level section - neither is a
    # requirement ID (app/ingestion/structure.py: sections are 2-level,
    # requirements are 3+ level).
    assert extract_requirement_id_candidates("PCI DSS v4.0.1 раздел 8.4") == []


def test_preserves_first_occurrence_order_and_dedupes():
    assert extract_requirement_id_candidates("8.4.2 и 8.3.1, снова 8.4.2") == ["8.4.2", "8.3.1"]


def test_no_matches_returns_empty_list():
    assert extract_requirement_id_candidates("просто общий вопрос без номеров") == []


def test_requirement_family_prefix_takes_first_two_segments():
    assert requirement_family_prefix("8.3.6") == "8.3"
    assert requirement_family_prefix("9.5.1.2") == "9.5"


def test_requirement_family_prefix_short_id_unchanged():
    assert requirement_family_prefix("8.3") == "8.3"
