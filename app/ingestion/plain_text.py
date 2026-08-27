"""Plain-text evidence parsing (PLAN.md section 3, Phase 2): interview
transcripts and config exports (.txt/.json/.csv) arrive as raw text
already, so unlike the PDF/.docx paths there is no structure to extract.
JSON is pretty-printed for readability before chunking (falling back to the
raw text if it doesn't parse); everything else is read as-is. Deliberately
lightweight rather than schema-aware, per the project's accuracy/speed
tradeoff for evidence formats (arbitrary, high-volume) vs. the Standard.

Encoding is auto-detected (`decode_best_effort`), not assumed to be UTF-8 -
the same Windows-console-output concern as `app/ingestion/archive.py`
applies to a standalone .txt/.csv export just as much as to one bundled in
an archive.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.encoding import decode_best_effort


class PlainTextParser:
    def __init__(self, path: Path) -> None:
        self.path = path

    def extract_text(self) -> str:
        raw = decode_best_effort(self.path.read_bytes())
        if self.path.suffix.lower() == ".json":
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return raw
        return raw

    def guess_title(self) -> str:
        return self.path.stem
