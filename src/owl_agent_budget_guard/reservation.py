"""Reservations: budget admission, usage recording and settlement.

A reservation is created by :meth:`Session.reserve` (or the
:meth:`Session.operation` context manager) after the guard has confirmed that
the estimated cost and declared calls fit within every applicable limit. The
estimate is held as reserved spending until the reservation is settled.

Settlement happens exactly once, through one of:

* :meth:`Reservation.reconcile` - explicit actual cost, state becomes ``settled``.
* :meth:`Reservation.release` - the operation is abandoned, state becomes ``released``.
* leaving an ``operation()`` block - automatic, using the rules in ``_settle``.
"""

from __future__ import annotations

import secrets
import time
from contextlib import AbstractContextManager
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from owl_agent_budget_guard.exceptions import (
    InvalidValueError,
    LimitExceededError,
    OwlBudgetError,
    ReservationError,
    ReservationOverrunError,
    UnknownCostError,
)
from owl_agent_budget_guard.ledger import COST_SOURCES, Scope
from owl_agent_budget_guard.money import ZERO, to_money
from owl_agent_budget_guard.validation import optional_count

if TYPE_CHECKING:
    from owl_agent_budget_guard.session import Session


class Reservation:
    """Budget held for one operation, plus the usage recorded against it.

    Attributes:
        id: Unique identifier, for logs and reports.
        estimated_cost: The amount reserved at admission.
        llm_calls: Declared maximum number of model requests.
        tool_calls: Declared maximum number of tool executions.
        state: ``"open"``, ``"settled"`` or ``"released"``.
    """

    def __init__(
        self,
        session: Session,
        reservation_id: str,
        estimated_cost: Decimal,
        llm_calls: int,
        tool_calls: int,
    ) -> None:
        self._session = session
        self.id = reservation_id
        self.estimated_cost = estimated_cost
        self.llm_calls = llm_calls
        self.tool_calls = tool_calls
        self.state = "open"
        self._started_at = time.monotonic()
        self._llm_recorded = 0
        self._tool_recorded = 0
        self._known_cost: dict[str, Decimal] = {}
        self._unknown_cost = False

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def _mutation(self) -> AbstractContextManager[None]:
        """Lock, check persistence health, and persist the session when the block ends."""
        return self._session._guard._mutation(self._session)

    def to_payload(self, now: float) -> dict[str, Any]:
        """Return the reservation's state in a form the store can persist."""
        return {
            "session_id": self._session.session_id,
            "estimated_cost": str(self.estimated_cost),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "llm_recorded": self._llm_recorded,
            "tool_recorded": self._tool_recorded,
            "known_cost": {source: str(amount) for source, amount in self._known_cost.items()},
            "unknown_cost": self._unknown_cost,
            "elapsed_seconds": max(0.0, now - self._started_at),
        }

    # ------------------------------------------------------------------
    # Usage recording
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        *,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = 0,
        reported_cost: object = None,
    ) -> None:
        """Record one completed model call.

        Args:
            model: Model identifier. Used to calculate cost from registered pricing.
            input_tokens: Input tokens reported by the provider, or ``None`` if unknown.
            output_tokens: Output tokens reported by the provider, or ``None`` if unknown.
            cached_input_tokens: Subset of ``input_tokens`` served from cache.
            reported_cost: Provider-reported cost. Takes precedence over calculated cost.

        Raises:
            InvalidValueError: For malformed arguments. Nothing is recorded.
            ReservationError: If the reservation is no longer open.
            LimitExceededError: If more calls are recorded than were declared, or a
                token limit is exceeded. The usage is recorded before this is raised.
            UnknownCostError: If the cost cannot be determined and ``unknown_cost``
                is ``"block"``.
            ExecutionTimeExceededError: If the session's time limit has passed.
        """
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise InvalidValueError("model must be a non-empty string when provided")
        input_count = optional_count(input_tokens, "input_tokens")
        output_count = optional_count(output_tokens, "output_tokens")
        cached_count = optional_count(cached_input_tokens, "cached_input_tokens") or 0
        if input_count is not None and cached_count > input_count:
            raise InvalidValueError("cached_input_tokens must not exceed input_tokens")
        reported = None if reported_cost is None else to_money(reported_cost, "reported_cost")

        session = self._session
        with self._mutation():
            self._require_open()
            session._check_deadline()
            if self._llm_recorded >= self.llm_calls:
                attempted = self._llm_recorded + 1
                session._guard._record_event(
                    scope=session._scope.name,
                    kind="limit_rejected",
                    detail=f"{attempted} model call(s) recorded but only {self.llm_calls} declared",
                )
                raise LimitExceededError(
                    f"reservation {self.id} declared {self.llm_calls} model call(s); "
                    f"recording call {attempted} is not allowed. Declare it with llm_calls=...",
                    scope=session._scope.name,
                    limit_name="llm_calls (per operation)",
                    limit=self.llm_calls,
                    current=attempted,
                )
            self._llm_recorded += 1
            session._move_request_to_used(llm_calls=1, tool_calls=0)
            session._add_tokens(input_count, output_count, cached_count)

            usage_complete = input_count is not None and output_count is not None
            pending_unknown: list[tuple[str, str]] = []
            if not usage_complete and session._token_limits_configured():
                pending_unknown.append(
                    (
                        "unreported_usage",
                        "token usage was not fully reported, so token limits cannot be enforced",
                    )
                )

            cost, source = self._price_call(
                model, input_count, output_count, cached_count, reported
            )
            if cost is None or source is None:
                self._unknown_cost = True
                pending_unknown.append(
                    (
                        "unknown_cost",
                        f"cost of model call is unknown (model={model!r}); pass reported_cost=... "
                        "or register pricing for this model",
                    )
                )
            else:
                self._add_known_cost(source, cost)

            # Token limits are enforced after the fact; the usage has already happened.
            session._enforce_token_limits(input_count, output_count)
            for kind, reason in pending_unknown:
                self._apply_unknown_policy(kind, reason)

    def record_tool_call(self, *, cost: object = None) -> None:
        """Record one completed tool execution.

        ``cost`` must be given explicitly. Pass ``0`` for a tool that is free.
        A missing cost is never assumed to be zero.
        """
        reported = None if cost is None else to_money(cost, "cost")
        session = self._session
        with self._mutation():
            self._require_open()
            session._check_deadline()
            if self._tool_recorded >= self.tool_calls:
                attempted = self._tool_recorded + 1
                session._guard._record_event(
                    scope=session._scope.name,
                    kind="limit_rejected",
                    detail=f"{attempted} tool call(s) recorded but only {self.tool_calls} declared",
                )
                raise LimitExceededError(
                    f"reservation {self.id} declared {self.tool_calls} tool call(s); "
                    f"recording call {attempted} is not allowed. Declare it with tool_calls=...",
                    scope=session._scope.name,
                    limit_name="tool_calls (per operation)",
                    limit=self.tool_calls,
                    current=attempted,
                )
            self._tool_recorded += 1
            session._move_request_to_used(llm_calls=0, tool_calls=1)
            if reported is None:
                self._unknown_cost = True
                self._apply_unknown_policy(
                    "unknown_cost",
                    "tool call cost was not provided; pass cost=... (use 0 for a free tool)",
                )
            else:
                self._add_known_cost("reported", reported)

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def reconcile(self, actual_cost: object = None) -> None:
        """Settle the reservation with the actual cost of the operation.

        Args:
            actual_cost: Total actual cost. If omitted, the sum of recorded costs
                is charged. It must not be lower than the cost already recorded.

        Raises:
            ReservationError: If the reservation was already settled or released.
            ReservationOverrunError: If the actual cost exceeds the reservation.
                The actual cost is recorded before this is raised.
        """
        explicit = None if actual_cost is None else to_money(actual_cost, "actual_cost")
        with self._mutation():
            self._require_open()
            if explicit is not None and explicit < self._known_total():
                raise InvalidValueError(
                    f"actual_cost {explicit} is lower than the {self._known_total()} already "
                    "recorded for this operation"
                )
            self._settle(success=True, explicit_total=explicit, raise_deferred=True)

    def release(self) -> None:
        """Abandon the operation and release its unused reservation.

        Costs already recorded are still charged, because they were incurred.
        Only the unused part of the reservation is released. If nothing was
        recorded, nothing is charged.
        """
        with self._mutation():
            self._require_open()
            self._settle(success=False, explicit_total=None, raise_deferred=True)

    def _finish_context(self, *, failed: bool) -> None:
        """Settle at the end of an ``operation()`` block. Never masks the block's own error."""
        with self._mutation():
            if not self.is_open:
                return
            self._settle(success=not failed, explicit_total=None, raise_deferred=not failed)

    def _settle(
        self, *, success: bool, explicit_total: Decimal | None, raise_deferred: bool
    ) -> None:
        """Apply the reservation to every scope. Caller must hold the lock and check state."""
        session = self._session
        guard = session._guard
        scopes = session._scopes()
        known_total = self._known_total()
        records = self._llm_recorded + self._tool_recorded
        breakdown: dict[str, Decimal] = dict(self._known_cost)
        deferred: list[OwlBudgetError] = []

        if explicit_total is not None:
            charged = explicit_total
            residual = explicit_total - known_total
            if residual > ZERO:
                breakdown["reconciled"] = residual
        else:
            has_unknown = self._unknown_cost or (success and records == 0)
            if has_unknown:
                # Never assume zero. Charge the larger of what is known and the estimate.
                charged = max(known_total, self.estimated_cost)
                fallback = charged - known_total
                if fallback > ZERO:
                    breakdown["estimated_fallback"] = fallback
                if success and records == 0:
                    guard._record_event(
                        scope=session._scope.name,
                        kind="unreported_usage",
                        detail=f"operation {self.id} completed without recording cost or usage",
                    )
                    if guard.policies.unknown_cost == "block":
                        deferred.append(
                            UnknownCostError(
                                "operation completed without recording any cost or usage. "
                                "Call record_llm_call()/record_tool_call() or reconcile("
                                "actual_cost=...) (use 0 for free work). The full estimate "
                                f"({self.estimated_cost}) was charged."
                            )
                        )
            else:
                charged = known_total

        estimated = self.estimated_cost
        apply_settlement(
            scopes,
            estimated=estimated,
            charged=charged,
            breakdown=breakdown,
            llm_calls=self.llm_calls,
            llm_recorded=self._llm_recorded,
            tool_calls=self.tool_calls,
            tool_recorded=self._tool_recorded,
            success=success,
            elapsed=time.monotonic() - self._started_at,
        )

        if charged > estimated:
            overrun = charged - estimated
            guard._record_event(
                scope=session._scope.name,
                kind="reservation_overrun",
                detail=f"operation {self.id} cost {charged} exceeded its reservation "
                f"{estimated} by {overrun}",
            )
            deferred.insert(
                0,
                ReservationOverrunError(
                    f"operation {self.id} cost {charged} but only {estimated} was reserved; "
                    "the actual cost has been recorded and the budget may now be exhausted"
                ),
            )

        for scope in scopes:
            remaining = scope.remaining_cost()
            if remaining is not None and remaining <= ZERO:
                guard._record_event(
                    scope=scope.name,
                    kind="budget_exhausted",
                    detail=f"remaining available budget is {remaining}",
                )

        if success:
            violation = session._find_time_violation(time.monotonic())
            if violation is not None:
                deferred.append(session._time_limit_error(*violation))

        self.state = "settled" if success else "released"
        session._open_reservations.pop(self.id, None)

        if deferred and raise_deferred:
            raise deferred[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self.is_open:
            raise ReservationError(
                f"reservation {self.id} is already {self.state}; it cannot be "
                "recorded, reconciled or released again"
            )

    def _known_total(self) -> Decimal:
        return sum(self._known_cost.values(), ZERO)

    def _add_known_cost(self, source: str, amount: Decimal) -> None:
        self._known_cost[source] = self._known_cost.get(source, ZERO) + amount

    def _price_call(
        self,
        model: str | None,
        input_count: int | None,
        output_count: int | None,
        cached_count: int,
        reported: Decimal | None,
    ) -> tuple[Decimal | None, str | None]:
        if reported is not None:
            return reported, "reported"
        if model is None or input_count is None or output_count is None:
            return None, None
        session = self._session
        guard = session._guard
        if model not in guard._pricing:
            return None, None
        cost = guard._pricing.calculate_cost(
            model,
            input_tokens=input_count,
            output_tokens=output_count,
            cached_input_tokens=cached_count,
            currency=guard.currency,
        )
        return cost, "calculated"

    def _apply_unknown_policy(self, kind: str, reason: str) -> None:
        guard = self._session._guard
        guard._record_event(
            scope=self._session._scope.name,
            kind=kind,
            detail=f"{reason} (operation {self.id})",
        )
        if guard.policies.unknown_cost == "block":
            raise UnknownCostError(
                f"{reason}. unknown_cost=block rejects this; supply the value, register "
                "pricing, or set unknown_cost to 'estimate' to charge the reservation estimate."
            )


def new_reservation_id() -> str:
    """Return a random, unguessable reservation identifier."""
    return f"res_{secrets.token_hex(8)}"


def apply_settlement(
    scopes: list[Scope],
    *,
    estimated: Decimal,
    charged: Decimal,
    breakdown: dict[str, Decimal],
    llm_calls: int,
    llm_recorded: int,
    tool_calls: int,
    tool_recorded: int,
    success: bool,
    elapsed: float,
) -> None:
    """Move one reservation from reserved to actual in every scope.

    Shared by normal settlement and by crash recovery, so both follow the same
    accounting rules. Caller must hold the guard lock.
    """
    for scope in scopes:
        scope.reserved_cost -= estimated
        scope.actual_cost += charged
        for source in COST_SOURCES:
            amount = breakdown.get(source)
            if amount:
                scope.cost_by_source[source] += amount

        unused_llm = llm_calls - llm_recorded
        scope.llm_calls_reserved -= unused_llm
        if not success:
            scope.llm_calls_used += unused_llm  # conservative: may have been made

        unused_tools = tool_calls - tool_recorded
        scope.tool_calls_reserved -= unused_tools
        if not success:
            scope.tool_calls_used += unused_tools

        if success:
            scope.successful_operations += 1
        else:
            scope.failed_operations += 1
        scope.operations_seconds_total += elapsed
