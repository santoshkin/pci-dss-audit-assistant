from app.ingestion.encoding import decode_best_effort


def test_decodes_utf8():
    assert decode_best_effort("Интервью про MFA".encode("utf-8")) == "Интервью про MFA"


def test_decodes_cp1251_windows_console_output():
    # Real shape of the bug this module fixes: Windows tools like
    # auditpol/arp write cp1251, not UTF-8 - a blind UTF-8 decode would
    # turn this into mojibake.
    raw = "Политика аудита системы".encode("cp1251")
    assert decode_best_effort(raw) == "Политика аудита системы"


def test_decodes_utf16_with_bom():
    # PowerShell GPO HTML exports are observed to be utf-16 with BOM.
    raw = "<html><body>отчёт</body></html>".encode("utf-16")
    assert decode_best_effort(raw) == "<html><body>отчёт</body></html>"


def test_empty_bytes_do_not_crash():
    assert decode_best_effort(b"") == ""
