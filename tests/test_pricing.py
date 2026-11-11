from decimal import Decimal

import pytest

from owl_agent_budget_guard import (
    BudgetGuard,
    InvalidValueError,
    ModelPricing,
    PricingError,
    PricingRegistry,
)


def _pricing(**overrides) -> ModelPricing:
    values = {
        "model": "example-model",
        "input_per_million": Decimal("1.00"),
        "output_per_million": Decimal("5.00"),
        "currency": "USD",
    }
    values.update(overrides)
    return ModelPricing(**values)


def _registry() -> PricingRegistry:
    registry = PricingRegistry()
    registry.register(_pricing())
    return registry


def test_input_cost_only() -> None:
    cost = _registry().calculate_cost(
        "example-model", input_tokens=1_000_000, output_tokens=0, currency="USD"
    )
    assert cost == Decimal("1.00000000")


def test_output_cost_only() -> None:
    cost = _registry().calculate_cost(
        "example-model", input_tokens=0, output_tokens=2_000, currency="USD"
    )
    assert cost == Decimal("0.01000000")


def test_combined_cost_is_exact() -> None:
    # 1000 * 1.00 / 1e6 + 500 * 5.00 / 1e6 = 0.001 + 0.0025 = 0.0035
    cost = _registry().calculate_cost(
        "example-model", input_tokens=1000, output_tokens=500, currency="USD"
    )
    assert cost == Decimal("0.00350000")


def test_cached_tokens_are_a_subset_of_input_and_use_cached_rate() -> None:
    registry = PricingRegistry()
    registry.register(_pricing(cached_input_per_million=Decimal("0.10")))
    # 800 uncached at 1.00 + 200 cached at 0.10 = 0.0008 + 0.00002 = 0.00082
    cost = registry.calculate_cost(
        "example-model",
        input_tokens=1000,
        output_tokens=0,
        cached_input_tokens=200,
        currency="USD",
    )
    assert cost == Decimal("0.00082000")


def test_rounding_is_up_so_costs_are_never_under_reported() -> None:
    # 1 token at 0.075 per million is 0.000000075, which must round up to 8 places.
    registry = PricingRegistry()
    registry.register(_pricing(input_per_million=Decimal("0.075"), output_per_million=Decimal("0")))
    cost = registry.calculate_cost("example-model", input_tokens=1, output_tokens=0, currency="USD")
    assert cost == Decimal("0.00000008")


def test_large_values_keep_decimal_precision() -> None:
    cost = _registry().calculate_cost(
        "example-model", input_tokens=123_456_789, output_tokens=987_654_321, currency="USD"
    )
    # 123456789 * 1 / 1e6 + 987654321 * 5 / 1e6 = 123.456789 + 4938.271605 = 5061.728394
    assert cost == Decimal("5061.72839400")


def test_unknown_model_raises_pricing_error() -> None:
    with pytest.raises(PricingError, match="no pricing registered for model 'missing'"):
        _registry().calculate_cost("missing", input_tokens=1, output_tokens=1, currency="USD")


def test_currency_mismatch_raises() -> None:
    with pytest.raises(PricingError, match="budget currency is EUR"):
        _registry().calculate_cost("example-model", input_tokens=1, output_tokens=1, currency="EUR")


def test_cached_tokens_without_cached_price_raise() -> None:
    with pytest.raises(PricingError, match="no cached_input_per_million"):
        _registry().calculate_cost(
            "example-model", input_tokens=10, output_tokens=0, cached_input_tokens=5, currency="USD"
        )


def test_cached_tokens_cannot_exceed_input() -> None:
    registry = PricingRegistry()
    registry.register(_pricing(cached_input_per_million=Decimal("0.1")))
    with pytest.raises(InvalidValueError, match="must not exceed input_tokens"):
        registry.calculate_cost(
            "example-model",
            input_tokens=10,
            output_tokens=0,
            cached_input_tokens=11,
            currency="USD",
        )


@pytest.mark.parametrize("bad_tokens", [-1, 1.5, True, "10"])
def test_invalid_token_counts_are_rejected(bad_tokens: object) -> None:
    with pytest.raises(InvalidValueError):
        _registry().calculate_cost(
            "example-model", input_tokens=bad_tokens, output_tokens=0, currency="USD"
        )


def test_duplicate_registration_requires_replace() -> None:
    registry = _registry()
    with pytest.raises(PricingError, match="already registered"):
        registry.register(_pricing())
    registry.register(_pricing(output_per_million=Decimal("6")), replace=True)
    assert registry.get("example-model").output_per_million == Decimal("6.00000000")


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_per_million": Decimal("-1")},
        {"output_per_million": "abc"},
        {"currency": "usd"},
        {"currency": "DOLLARS"},
        {"model": "   "},
        {"model": "bad model!"},
        {"cached_input_per_million": Decimal("-0.5")},
        {"source": "x" * 501},
    ],
)
def test_invalid_pricing_is_rejected(overrides: dict) -> None:
    with pytest.raises(InvalidValueError):
        _pricing(**overrides)


def test_guard_rejects_pricing_in_other_currency() -> None:
    with pytest.raises(PricingError, match="this guard uses USD"):
        BudgetGuard(currency="USD", pricing=[_pricing(currency="EUR")])


def test_all_returns_sorted_entries() -> None:
    registry = PricingRegistry()
    registry.register(_pricing(model="zeta"))
    registry.register(_pricing(model="alpha"))
    assert [p.model for p in registry.all()] == ["alpha", "zeta"]
    assert "alpha" in registry
    assert "beta" not in registry
