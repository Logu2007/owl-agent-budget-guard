from decimal import Decimal

import pytest

from owl_agent_budget_guard import InvalidValueError
from owl_agent_budget_guard.money import format_money, to_money


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.10", Decimal("0.10000000")),
        (5, Decimal("5.00000000")),
        (Decimal("2.5"), Decimal("2.50000000")),
        (0.1, Decimal("0.10000000")),  # float goes through its repr, not binary
        ("  1.25 ", Decimal("1.25000000")),
        ("0.100000000", Decimal("0.10000000")),  # trailing zeros are not extra precision
        (0, Decimal("0.00000000")),
    ],
)
def test_accepted_values_are_quantized_decimals(value: object, expected: Decimal) -> None:
    assert to_money(value) == expected
    assert isinstance(to_money(value), Decimal)


def test_float_sum_is_exact_after_conversion() -> None:
    total = to_money(0.1) + to_money(0.2)
    assert total == Decimal("0.30000000")


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        False,
        -1,
        "-0.01",
        float("nan"),
        float("inf"),
        "NaN",
        "abc",
        "",
        "0.000000001",  # nine decimal places
        None,
        [1],
        "1e100000",
    ],
)
def test_invalid_money_values_are_rejected(bad_value: object) -> None:
    with pytest.raises(InvalidValueError):
        to_money(bad_value, "amount")


def test_error_message_names_the_field() -> None:
    with pytest.raises(InvalidValueError, match="estimated_cost must not be negative"):
        to_money(-1, "estimated_cost")


def test_format_money_is_compact() -> None:
    assert format_money(Decimal("0.10000000"), "USD") == "0.1 USD"
    assert format_money(Decimal("0E-8"), "USD") == "0 USD"
    assert format_money(Decimal("5.00000000"), "EUR") == "5 EUR"
