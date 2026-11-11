"""Shared fixtures and helpers for the test suite.

Tests use a fake clock for time-limit behaviour so that they are deterministic
and never sleep.
"""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from owl_agent_budget_guard import BudgetGuard, Limits, ModelPricing, Policies

# Test-only prices. They are not provider prices and are not bundled with the package.
TEST_MODEL = "test-model"
TEST_PRICING = ModelPricing(
    model=TEST_MODEL,
    input_per_million=Decimal("1.00"),
    output_per_million=Decimal("5.00"),
    cached_input_per_million=Decimal("0.10"),
    currency="USD",
    source="test fixture",
)


class FakeClock:
    """Replacement for ``time.monotonic`` that only moves when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("owl_agent_budget_guard.guard.time.monotonic", fake)
    monkeypatch.setattr("owl_agent_budget_guard.session.time.monotonic", fake)
    monkeypatch.setattr("owl_agent_budget_guard.reservation.time.monotonic", fake)
    return fake


@pytest.fixture
def make_guard():
    """Factory for guards with the test pricing registered."""

    def factory(
        *,
        limits: Limits | None = None,
        policies: Policies | None = None,
        pricing: bool = True,
        agent_name: str = "test-agent",
        store=None,
    ) -> BudgetGuard:
        return BudgetGuard(
            agent_name=agent_name,
            currency="USD",
            limits=limits,
            policies=policies,
            pricing=[TEST_PRICING] if pricing else [],
            store=store,
        )

    return factory


_counter = itertools.count()


@pytest.fixture
def unique_name() -> str:
    return f"task {next(_counter)}"
