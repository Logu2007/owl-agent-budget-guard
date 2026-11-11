"""Monetary value handling.

All money is represented as :class:`decimal.Decimal`. Floats are accepted as a
convenience and converted through their shortest ``repr``, so ``0.1`` becomes
``Decimal("0.1")`` instead of the binary approximation.

Amounts have at most eight decimal places. Values with more places are
rejected rather than silently rounded.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, localcontext

from owl_agent_budget_guard.exceptions import InvalidValueError

MONEY_PLACES = 8
MONEY_QUANTUM = Decimal("1E-8")
ZERO = Decimal("0")
_WORKING_PRECISION = 60


def _coerce_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise InvalidValueError(f"{field} must be a number, not a boolean")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidValueError(f"{field} must be a finite number")
        return Decimal(repr(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation as exc:
            raise InvalidValueError(f"{field} is not a valid decimal number") from exc
    raise InvalidValueError(
        f"{field} must be a number or a decimal string, not {type(value).__name__}"
    )


def to_money(value: object, field: str = "amount") -> Decimal:
    """Convert a user-supplied amount into a non-negative, 8-place Decimal.

    Raises :class:`InvalidValueError` for booleans, non-finite values, negative
    values, values with more than eight decimal places, and values too large to
    represent safely.
    """
    number = _coerce_decimal(value, field)
    if not number.is_finite():
        raise InvalidValueError(f"{field} must be a finite number")
    if number < ZERO:
        raise InvalidValueError(f"{field} must not be negative")
    with localcontext() as context:
        context.prec = _WORKING_PRECISION
        try:
            quantized = number.quantize(MONEY_QUANTUM)
        except InvalidOperation as exc:
            raise InvalidValueError(f"{field} is too large") from exc
        if quantized != number:
            raise InvalidValueError(f"{field} must have at most {MONEY_PLACES} decimal places")
    return quantized


def format_money(amount: Decimal, currency: str) -> str:
    """Format an amount for terminal output, e.g. ``"0.1 USD"``."""
    return f"{format(amount.normalize(), 'f')} {currency}"
