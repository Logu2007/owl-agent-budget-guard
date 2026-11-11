"""Accounting state for one budget scope (the agent, or one session).

Scopes are plain data. Every read and write must happen while holding the
guard's lock. The helpers here do not lock by themselves, because a decision
often spans several scopes and must be atomic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from owl_agent_budget_guard.limits import Limits
from owl_agent_budget_guard.money import ZERO

# How the amount of a settled operation was determined.
#   reported:          supplied by the caller from provider or tool usage data
#   calculated:        computed locally from registered pricing and token counts
#   reconciled:        residual from an explicit reconcile(actual_cost=...) call
#   estimated_fallback: no cost was known; the reservation estimate was charged
COST_SOURCES: tuple[str, ...] = ("reported", "calculated", "reconciled", "estimated_fallback")


def _zero_costs() -> dict[str, Decimal]:
    return {source: ZERO for source in COST_SOURCES}


@dataclass
class Scope:
    """Mutable accounting counters for one scope."""

    name: str
    limits: Limits
    started_at: float
    actual_cost: Decimal = ZERO
    reserved_cost: Decimal = ZERO
    cost_by_source: dict[str, Decimal] = field(default_factory=_zero_costs)
    llm_calls_used: int = 0
    llm_calls_reserved: int = 0
    tool_calls_used: int = 0
    tool_calls_reserved: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    operations_seconds_total: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started_at)

    def remaining_cost(self) -> Decimal | None:
        """Available budget: limit minus actual spending minus reserved spending."""
        if self.limits.max_cost is None:
            return None
        return self.limits.max_cost - self.actual_cost - self.reserved_cost

    def remaining_execution_time(self, now: float) -> float | None:
        if self.limits.max_execution_time is None:
            return None
        return max(0.0, self.limits.max_execution_time - self.elapsed(now))

    def time_exceeded(self, now: float) -> bool:
        limit = self.limits.max_execution_time
        return limit is not None and self.elapsed(now) >= limit


def find_cost_violation(scopes: list[Scope], amount: Decimal) -> tuple[Scope, Decimal] | None:
    """Return ``(scope, available)`` for the first scope that cannot cover ``amount``.

    A scope with no remaining budget (zero or negative) rejects every amount,
    including zero. Budget exhaustion therefore stops even free operations.
    """
    for scope in scopes:
        remaining = scope.remaining_cost()
        if remaining is None:
            continue
        if remaining <= ZERO or amount > remaining:
            return scope, remaining
    return None


def find_request_violation(
    scopes: list[Scope], llm_calls: int, tool_calls: int
) -> tuple[Scope, str, int, int] | None:
    """Return ``(scope, limit_name, limit, current)`` if declared calls would exceed a limit."""
    for scope in scopes:
        limits = scope.limits
        if limits.max_llm_calls is not None:
            current = scope.llm_calls_used + scope.llm_calls_reserved
            if current + llm_calls > limits.max_llm_calls:
                return scope, "max_llm_calls", limits.max_llm_calls, current
        if limits.max_tool_calls is not None:
            current = scope.tool_calls_used + scope.tool_calls_reserved
            if current + tool_calls > limits.max_tool_calls:
                return scope, "max_tool_calls", limits.max_tool_calls, current
    return None


def find_token_exhaustion(scopes: list[Scope]) -> tuple[Scope, int, int] | None:
    """Return ``(scope, limit, current)`` when a scope's total-token limit is reached."""
    for scope in scopes:
        limit = scope.limits.max_tokens
        if limit is not None and scope.total_tokens >= limit:
            return scope, limit, scope.total_tokens
    return None
