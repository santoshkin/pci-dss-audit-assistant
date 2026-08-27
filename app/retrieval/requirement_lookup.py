"""Exact Requirement ID detection in a user query (PLAN.md section 9): if
the query names a specific ID, an exact metadata lookup should run before
semantic search, so an imperfect embedding match can't push the actual
requirement out of the results.
"""

import re

# The optional prefix letter is uppercase-only, matching the real
# requirement ID convention (Appendix IDs like "A1.1.1" -
# app/ingestion/structure.py's REQ_LINE_RE uses the same [A-Z]?). This
# isn't just consistency: a *lowercase* "v" there would also match the
# start of a version number like "v4.0.1" (itself 3 dot-separated
# segments, indistinguishable by segment count alone from a real ID like
# "8.4.2") - restricting to uppercase means "v4.0.1" fails to match at
# either candidate start position ([A-Z]? can't consume lowercase "v", and
# \b doesn't hold between "v" and "4" since both are word characters), so
# the extremely common phrasing "PCI DSS v4.0.1" is never mistaken for a
# requirement ID.
REQUIREMENT_ID_RE = re.compile(r"\b([A-Z]?\d+\.\d+(?:\.\d+){1,3})\b")


def extract_requirement_id_candidates(query: str) -> list[str]:
    """Returns the distinct requirement-ID-shaped tokens in `query`, in
    order of first appearance. Requires at least 3 dot-separated segments
    (e.g. "8.4.2") to avoid matching plain section references like "8.4"
    (see app/ingestion/structure.py for why 2-level numbers are section
    titles, not requirements, in this document) and excludes "v"-prefixed
    version numbers like "v4.0.1" (see REQUIREMENT_ID_RE)."""
    seen: dict[str, None] = {}
    for match in REQUIREMENT_ID_RE.finditer(query):
        seen.setdefault(match.group(1), None)
    return list(seen)


def requirement_family_prefix(requirement_id: str) -> str:
    """The 2-level section grouping a requirement ID belongs to (e.g.
    "8.3.6" -> "8.3") - matches app/ingestion/structure.py's own
    convention that 2-level numbers are section titles, not individual
    testable requirements, so this is the natural "sibling group" unit:
    a broad question that surfaces "8.3.6" almost always also needs
    "8.3.4"/"8.3.7"/"8.3.9"/... which an embedding-similarity pass alone
    won't reliably all rank highly for the same query (see
    app/services/project_chat.py, ARCHITECTURE.md section 21)."""
    parts = requirement_id.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else requirement_id
