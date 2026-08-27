from app.services.evidence_intake import _archive_format, _base_name


def test_archive_format_detects_zip():
    assert _archive_format("DESKTOP-ASVDD (2).zip") == "zip"


def test_archive_format_detects_tar_gz():
    assert _archive_format("DESKTOP-ASVDD.tar.gz") == "tar"


def test_archive_format_detects_tgz():
    assert _archive_format("evidence.tgz") == "tar"


def test_archive_format_none_for_regular_file():
    assert _archive_format("interview.txt") is None
    assert _archive_format("policy.docx") is None


def test_base_name_strips_double_suffix_for_tar_gz():
    # Path(...).stem alone would leave a stray ".tar" here.
    assert _base_name("DESKTOP-ASVDD.tar.gz") == "DESKTOP-ASVDD"


def test_base_name_regular_files():
    assert _base_name("interview.txt") == "interview"
    assert _base_name("DESKTOP-ASVDD (2).zip") == "DESKTOP-ASVDD (2)"
