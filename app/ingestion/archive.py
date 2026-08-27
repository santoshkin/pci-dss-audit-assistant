"""Parses technical-assessment evidence archives (PLAN.md section 3, Phase
2 - "выгрузки конфигураций"): bundles of system config/log dumps collected
during technical checks, packaged as .zip (Windows: GPO reports, secedit,
auditpol, netstat, ...) or .tar.gz (Unix: sshd_config, sudoers,
pwquality.conf, ...). Sensitive data (password hashes, etc.) is removed by
the assessor before the archive is produced - not this project's concern.

Each member is read as plain text (config/log/txt/html - no attempt to
strip HTML markup, same lightweight-over-precise tradeoff as
`PlainTextParser`) and kept as a separate (name, text) pair, not
concatenated, so `GenericChunker.chunk_archive` can tag each resulting
chunk's `section` with the originating file name - citations point at
"secedit.txt", not just "the archive".

Windows console tools (`auditpol`, `arp`, `netsh`, PowerShell HTML
exports) write output in the OS locale's codepage (observed: cp1251,
cp1125/cp866, utf-16 with BOM for GPO HTML reports) - a blind UTF-8 decode
turns Cyrillic output into mojibake, which would poison embeddings for
exactly the evidence this format exists for. `charset_normalizer`
(already an installed transitive dependency of `requests`/`httpx`, pinned
directly here since this module now relies on it as more than that)
detects the actual encoding per member instead.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from app.ingestion.encoding import decode_best_effort


class ArchiveParser:
    def __init__(self, path: Path, archive_format: str) -> None:
        if archive_format not in ("zip", "tar"):
            raise ValueError(f"Unsupported archive_format: {archive_format!r}")
        self.path = path
        self.archive_format = archive_format

    def extract_members(self) -> list[tuple[str, str]]:
        """Returns (member_name, text) pairs for every non-empty file
        member, decoded as UTF-8 (errors replaced). Directory entries and
        empty files are skipped; nothing here detects/rejects binary
        content beyond a best-effort UTF-8 decode."""
        if self.archive_format == "zip":
            return self._extract_zip()
        return self._extract_tar()

    def _extract_tar(self) -> list[tuple[str, str]]:
        members: list[tuple[str, str]] = []
        with tarfile.open(self.path, mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.size == 0:
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                text = decode_best_effort(fileobj.read()).strip()
                if text:
                    members.append((member.name, text))
        return members

    def _extract_zip(self) -> list[tuple[str, str]]:
        members: list[tuple[str, str]] = []
        with zipfile.ZipFile(self.path) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.file_size == 0:
                    continue
                text = decode_best_effort(zf.read(info.filename)).strip()
                if text:
                    members.append((info.filename, text))
        return members
