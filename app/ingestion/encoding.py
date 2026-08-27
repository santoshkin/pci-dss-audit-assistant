"""Shared text-decoding helper for ingestion paths that read raw bytes
from an arbitrary-origin file (evidence archives, plain-text evidence) -
see `app/ingestion/archive.py`'s module docstring for why a blind UTF-8
decode isn't safe here (Windows console tools commonly write cp1251/
cp1125/utf-16, not UTF-8).
"""

from __future__ import annotations

from charset_normalizer import from_bytes


def decode_best_effort(raw: bytes) -> str:
    result = from_bytes(raw).best()
    if result is None:
        return raw.decode("utf-8", errors="replace")
    return str(result)
