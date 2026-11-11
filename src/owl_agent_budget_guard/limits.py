"""Limit and policy settings.

``None`` for any limit means "not limited by this guard". Limits are never
inferred or increased automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from owl_agent_budget_guard.exceptions import InvalidValueError
from owl_agent_budget_guard.money import to_money
from owl_agent_budget_guard.validation import optional_count

OnBudgetExceeded = Literal["stop", "raise", "pause"]
UnknownCostPolicy = Literal["block", "estimate"]

ON_BUDGET_EXCEEDED_VALUES: tuple[str, ...] = ("stop", "raise", "pause")
UNKNOWN_COST_VALUES: tuple[str, ...] = ("block", "estimate")


@dataclass(frozen=True)
class Limits:
    """Resource limits for one scope (the agent or one session).

    Attributes:
        max_cost: Maximum actual plus reserved spending, in the guard currency.
        max_operation_cost: Maximum estimated cost of a single operation.
        max_llm_calls: Maximum model requests (declared requests count when reserved).
        max_tool_calls: Maximum tool executions.
        max_tokens: Maximum input plus output tokens recorded in this scope.
        max_input_tokens_per_call: Maximum input tokens for one model call.
        max_output_tokens_per_call: Maximum output tokens for one model call.
        max_execution_time: Maximum monotonic elapsed seconds for this scope.
    """

    max_cost: Decimal | None = None
    max_operation_cost: Decimal | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_input_tokens_per_call: int | None = None
    max_output_tokens_per_call: int | None = None
    max_execution_time: float | None = None

    def __post_init__(self) -> None:
        if self.max_cost is not None:
            object.__setattr__(self, "max_cost", to_money(self.max_cost, "max_cost"))
        if self.max_operation_cost is not None:
            object.__setattr__(
                self,
                "max_operation_cost",
                to_money(self.max_operation_cost, "max_operation_cost"),
            )
        for name in (
            "max_llm_calls",
            "max_tool_calls",
            "max_tokens",
            "max_input_tokens_per_call",
            "max_output_tokens_per_call",
        ):
            object.__setattr__(self, name, optional_count(getattr(self, name), name))
        if self.max_execution_time is not None:
            seconds = self.max_execution_time
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise InvalidValueError("max_execution_time must be a positive number of seconds")
            object.__setattr__(self, "max_execution_time", float(seconds))


@dataclass(frozen=True)
class Policies:
    """What happens when a limit is hit.

    Attributes:
        on_budget_exceeded: ``"stop"`` rejects the operation and stops the
            session (all later operations are rejected). ``"raise"`` rejects the
            operation and keeps the session usable for smaller operations.
            ``"pause"`` rejects the operation with :class:`ApprovalRequiredError`
            and pauses the session.
        unknown_cost: ``"block"`` raises :class:`UnknownCostError` when a cost
            cannot be determined. ``"estimate"`` charges the reservation estimate
            conservatively and records an event instead of raising.
    """

    on_budget_exceeded: OnBudgetExceeded = "stop"
    unknown_cost: UnknownCostPolicy = "block"

    def __post_init__(self) -> None:
        if self.on_budget_exceeded not in ON_BUDGET_EXCEEDED_VALUES:
            raise InvalidValueError(
                f"on_budget_exceeded must be one of {', '.join(ON_BUDGET_EXCEEDED_VALUES)}"
            )
        if self.unknown_cost not in UNKNOWN_COST_VALUES:
            raise InvalidValueError(f"unknown_cost must be one of {', '.join(UNKNOWN_COST_VALUES)}")
