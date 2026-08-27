import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from app.ingestion import ArchiveParser


def test_extract_zip_members_decodes_mixed_encodings(tmp_path: Path):
    path = tmp_path / "evidence.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("secedit.txt", "MinimumPasswordLength = 0".encode("utf-8"))
        zf.writestr("audit.txt", "Политика аудита системы".encode("cp1251"))
        zf.writestr("empty.txt", b"")
        zf.writestr("adir/", b"")

    members = ArchiveParser(path, "zip").extract_members()

    assert dict(members) == {
        "secedit.txt": "MinimumPasswordLength = 0",
        "audit.txt": "Политика аудита системы",
    }


def test_extract_tar_members_decodes_and_skips_empty(tmp_path: Path):
    path = tmp_path / "evidence.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, content in [("sshd_config", b"PermitRootLogin no\n"), ("empty_file", b"")]:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    members = ArchiveParser(path, "tar").extract_members()

    assert members == [("sshd_config", "PermitRootLogin no")]


def test_unsupported_archive_format_raises(tmp_path: Path):
    path = tmp_path / "x.rar"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        ArchiveParser(path, "rar")
