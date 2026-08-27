import json

from app.ingestion.plain_text import PlainTextParser


def test_extract_text_plain_txt_is_read_verbatim(tmp_path):
    path = tmp_path / "interview.txt"
    path.write_text("Интервью: MFA обязательна для CDE.", encoding="utf-8")
    assert PlainTextParser(path).extract_text() == "Интервью: MFA обязательна для CDE."


def test_extract_text_valid_json_is_pretty_printed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"mfa_enabled": true, "rotation_days": 60}', encoding="utf-8")
    result = PlainTextParser(path).extract_text()
    assert result == json.dumps({"mfa_enabled": True, "rotation_days": 60}, ensure_ascii=False, indent=2)


def test_extract_text_invalid_json_falls_back_to_raw(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json,,,", encoding="utf-8")
    assert PlainTextParser(path).extract_text() == "{not valid json,,,"


def test_guess_title_is_the_filename_stem(tmp_path):
    path = tmp_path / "config_export.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert PlainTextParser(path).guess_title() == "config_export"
