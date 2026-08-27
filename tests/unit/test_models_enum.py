"""Regression test for a bug found and fixed during Phase 1 (see
ARCHITECTURE.md section 15): without `values_callable`, SQLAlchemy's
`Enum(PythonEnum)` stores the member NAME ("NON_COMPLIANT") as the
Postgres label instead of `.value` ("non_compliant"), diverging from the
API's JSON contract. `_pg_enum` is the fix - this locks its behavior in
for all four enum types so it can't silently regress.
"""

from app.db.models import (
    ComplianceStatus,
    EvidenceType,
    FindingStatus,
    ProjectStatus,
    UserRole,
    _pg_enum,
)


def test_pg_enum_uses_lowercase_values_not_member_names():
    assert _pg_enum(ComplianceStatus, "x").enums == [s.value for s in ComplianceStatus]
    assert _pg_enum(ProjectStatus, "x").enums == [s.value for s in ProjectStatus]
    assert _pg_enum(FindingStatus, "x").enums == [s.value for s in FindingStatus]
    assert _pg_enum(UserRole, "x").enums == [s.value for s in UserRole]
    assert _pg_enum(EvidenceType, "x").enums == [s.value for s in EvidenceType]


def test_pg_enum_values_are_lowercase_snake_case():
    for member in ComplianceStatus:
        assert member.value == member.value.lower()
        assert member.value != member.name
