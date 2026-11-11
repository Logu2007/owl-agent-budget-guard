"""Shared input validation helpers for names, currencies and counts."""

from __future__ import annotations

import re

from owl_agent_budget_guard.exceptions import InvalidValueError

# Used for agent, session and model names. Kept permissive but free of control
# characters so names are safe to print and to store in reports.
NAME_PATTERN_TEXT = r"[A-Za-z0-9][A-Za-z0-9 ._:/-]{0,99}"
CURRENCY_PATTERN_TEXT = r"[A-Z]{3}"

_NAME_RE = re.compile(NAME_PATTERN_TEXT)
_CURRENCY_RE = re.compile(CURRENCY_PATTERN_TEXT)


def validate_name(value: object, field: str) -> str:
    """Return ``value`` if it is a valid 1-100 character name."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise InvalidValueError(
            f"{field} must be 1-100 characters: letters, digits, spaces and . _ : / -, "
            "starting with a letter or digit"
        )
    return value


def validate_currency(value: object) -> str:
    """Return ``value`` if it is a three-letter upper-case currency code."""
    if not isinstance(value, str) or not _CURRENCY_RE.fullmatch(value):
        raise InvalidValueError("currency must be a three-letter upper-case code such as USD")
    return value


def require_count(value: object, field: str) -> int:
    """Return a non-negative integer; reject booleans, floats and negatives."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValueError(f"{field} must be an integer")
    if value < 0:
        raise InvalidValueError(f"{field} must not be negative")
    return value


def optional_count(value: object, field: str) -> int | None:
    """Like :func:`require_count`, but ``None`` means "not set"."""
    if value is None:
        return None
    return require_count(value, field)
