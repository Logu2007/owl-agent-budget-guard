"""Sessions: budget scopes for one task, which hand out reservations.

Each session owns its own accounting scope, so two sessions never share
counters. Every admission is also checked against the agent-wide scope held by
the :class:`~owl_agent_budget_guard.guard.BudgetGuard`.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NoReturn

from owl_agent_budget_guard.exceptions import (
    ApprovalRequiredError,
    BudgetExceededError,
    ExecutionTimeExceededError,
    LimitExceededError,
    SessionClosedError,
    SessionStoppedError,
)
from owl_agent_budget_guard.ledger import (
    Scope,
    find_cost_violation,
    find_request_violation,
    find_token_exhaustion,
)
from owl_agent_budget_guard.limits import Limits
from owl_agent_budget_guard.money import ZERO, to_money
from owl_agent_budget_guard.report import build_session_report
from owl_agent_budget_guard.reservation import Reservation, new_reservation_id
from owl_agent_budget_guard.store import SessionRow, limits_to_payload
from owl_agent_budget_guard.validation import require_count, validate_name

if TYPE_CHECKING:
    from owl_agent_budget_guard.guard import BudgetGuard

_SESSION_STATES: tuple[str, ...] = ("active", "stopped", "paused", "closed")


class Session:
    """A budget scope for one task, such as one agent run.

    Create sessions with :meth:`BudgetGuard.session`. Use them as context managers
    so that any reservation still open when the block ends is released.

    Session states:
        ``active``: operations are accepted.
        ``stopped``: a ``stop`` policy was triggered; all new operations are rejected.
        ``paused``: a ``pause`` policy was triggered; all new operations are rejected.
        ``closed``: :meth:`close` was called; all new operations are rejected.
    """

    def __init__(
        self,
        *,
        guard: BudgetGuard,
        name: str,
        limits: Limits,
        session_id: str | None = None,
        scope: Scope | None = None,
        state: str = "active",
    ) -> None:
        if not isinstance(limits, Limits):
            raise TypeError("limits must be a Limits instance")
        if state not in _SESSION_STATES:
            raise ValueError(f"unknown session state {state!r}")
        self._guard = guard
        self.name = validate_name(name, "session name")
        self.session_id = session_id if session_id is not None else f"ses_{secrets.token_hex(8)}"
        self._scope = (
            scope
            if scope is not None
            else Scope(
                name=f"session:{self.session_id}",
                limits=limits,
                started_at=time.monotonic(),
            )
        )
        self._state = state
        self._closed_at: float | None = None
        self._open_reservations: dict[str, Reservation] = {}
        guard._attach_session(self)

    def _row(self) -> SessionRow:
        return SessionRow(
            session_id=self.session_id,
            name=self.name,
            state=self._state,
            scope_name=self._scope.name,
            limits=limits_to_payload(self._scope.limits),
        )

    # ------------------------------------------------------------------
    # Context manager and lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False

    @property
    def state(self) -> str:
        return self._state

    def close(self) -> None:
        """Close the session and release any reservations still open.

        Open reservations are released as failed operations, so only costs that
        were already recorded are charged. Calling close() twice is harmless.
        """
        with self._guard._mutation(self):
            if self._state == "closed":
                return
            for reservation in list(self._open_reservations.values()):
                reservation._settle(success=False, explicit_total=None, raise_deferred=False)
            self._state = "closed"
            self._closed_at = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> None:
        """Raise if the session can no longer accept work. Does not reserve anything.

        Raises:
            SessionClosedError, SessionStoppedError, ApprovalRequiredError: session state.
            ExecutionTimeExceededError: the time limit has passed.
            LimitExceededError: the token or request limit is reached.
            BudgetExceededError: no budget remains (applies the ``on_budget_exceeded`` policy).
        """
        with self._guard._mutation(self):
            self._require_usable()
            self._check_deadline()
            self._check_token_exhaustion()
            self._check_request_exhaustion()
            violation = find_cost_violation(self._scopes(), ZERO)
            if violation is not None:
                scope, available = violation
                self._reject_for_budget(
                    scope,
                    requested=ZERO,
                    available=available,
                    reason="budget is exhausted",
                )

    def reserve(
        self,
        estimated_cost: object,
        *,
        llm_calls: int = 0,
        tool_calls: int = 0,
    ) -> Reservation:
        """Reserve ``estimated_cost`` and declared calls before running an operation.

        Admission is atomic across the agent and session scopes. The reservation
        is held until it is reconciled, released, or the session closes.

        Args:
            estimated_cost: Expected cost of the operation. It is required, so an
                operation can never be admitted without a cost estimate.
            llm_calls: Maximum model requests the operation may make.
            tool_calls: Maximum tool executions the operation may make.

        Raises:
            BudgetExceededError: insufficient budget, or the operation estimate is
                above ``max_operation_cost``. Under ``stop`` or ``pause`` the
                session also changes state.
            ApprovalRequiredError: the ``pause`` policy triggered.
            LimitExceededError: request or token limits would be exceeded.
            ExecutionTimeExceededError: the time limit has passed.
        """
        cost = to_money(estimated_cost, "estimated_cost")
        llm = require_count(llm_calls, "llm_calls")
        tools = require_count(tool_calls, "tool_calls")

        guard = self._guard
        with guard._mutation(self):
            self._require_usable()
            self._check_deadline()
            self._check_token_exhaustion()
            self._check_request_capacity(llm, tools)
            self._check_operation_cap(cost)
            violation = find_cost_violation(self._scopes(), cost)
            if violation is not None:
                scope, available = violation
                reason = (
                    "budget is exhausted"
                    if available <= ZERO
                    else f"operation estimate {cost} does not fit"
                )
                self._reject_for_budget(scope, requested=cost, available=available, reason=reason)

            reservation = Reservation(self, new_reservation_id(), cost, llm, tools)
            for scope in self._scopes():
                scope.reserved_cost += cost
                scope.llm_calls_reserved += llm
                scope.tool_calls_reserved += tools
            self._open_reservations[reservation.id] = reservation
            return reservation

    @contextmanager
    def operation(
        self,
        estimated_cost: object,
        *,
        llm_calls: int = 0,
        tool_calls: int = 0,
    ) -> Iterator[Reservation]:
        """Reserve budget for a block of work and settle it automatically.

        * Admission failure raises before the block runs.
        * If the block raises, recorded costs are charged, the rest is released,
          and the original exception propagates unchanged.
        * If the block completes, the reservation is settled with recorded costs.
          If none were recorded, the estimate is charged and, under
          ``unknown_cost="block"``, :class:`UnknownCostError` is raised.

        Example::

            with session.operation(estimated_cost="0.10", llm_calls=1) as op:
                answer = call_model()
                op.record_llm_call(model="my-model", input_tokens=900, output_tokens=120)
        """
        reservation = self.reserve(estimated_cost, llm_calls=llm_calls, tool_calls=tool_calls)
        try:
            yield reservation
        except BaseException:
            reservation._finish_context(failed=True)
            raise
        else:
            reservation._finish_context(failed=False)

    def report(self) -> dict[str, Any]:
        """Return a JSON-compatible usage report for this session and the agent."""
        guard = self._guard
        with guard._lock:
            agent_now = time.monotonic()
            session_now = self._closed_at if self._closed_at is not None else agent_now
            return build_session_report(
                agent_name=guard.agent_name,
                currency=guard.currency,
                session_id=self.session_id,
                session_name=self.name,
                session_state=self._state,
                agent_scope=guard._agent_scope,
                session_scope=self._scope,
                agent_now=agent_now,
                session_now=session_now,
                events=guard._events,
            )

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold the guard lock)
    # ------------------------------------------------------------------

    def _scopes(self) -> list[Scope]:
        return [self._guard._agent_scope, self._scope]

    def _require_usable(self) -> None:
        if self._state == "closed":
            raise SessionClosedError(f"session {self.session_id} is closed")
        if self._state == "stopped":
            raise SessionStoppedError(
                f"session {self.session_id} was stopped by the on_budget_exceeded=stop policy"
            )
        if self._state == "paused":
            error = ApprovalRequiredError(
                f"session {self.session_id} is paused and awaiting approval; "
                "create a new session with explicit limits to continue"
            )
            error.scope = self._scope.name
            raise error

    def _find_time_violation(self, now: float) -> tuple[Scope, float, float] | None:
        for scope in self._scopes():
            limit = scope.limits.max_execution_time
            if limit is not None and scope.elapsed(now) >= limit:
                return scope, limit, scope.elapsed(now)
        return None

    def _time_limit_error(
        self, scope: Scope, limit: float, elapsed: float
    ) -> ExecutionTimeExceededError:
        self._guard._record_event(
            scope=scope.name,
            kind="time_limit_exceeded",
            detail=f"elapsed {elapsed:.3f}s reached max_execution_time {limit}s",
        )
        return ExecutionTimeExceededError(
            f"{scope.name}: max_execution_time reached (elapsed {elapsed:.3f}s, limit {limit}s). "
            "Running work is not interrupted; no new work is accepted.",
            scope=scope.name,
            limit_name="max_execution_time",
            limit=limit,
            current=round(elapsed, 3),
        )

    def _check_deadline(self) -> None:
        violation = self._find_time_violation(time.monotonic())
        if violation is not None:
            raise self._time_limit_error(*violation)

    def _check_token_exhaustion(self) -> None:
        found = find_token_exhaustion(self._scopes())
        if found is not None:
            scope, limit, current = found
            self._limit_error(scope, "max_tokens", limit, current, "token limit reached")

    def _check_request_exhaustion(self) -> None:
        for scope in self._scopes():
            if scope.limits.max_llm_calls is not None:
                current = scope.llm_calls_used + scope.llm_calls_reserved
                if current >= scope.limits.max_llm_calls:
                    self._limit_error(
                        scope,
                        "max_llm_calls",
                        scope.limits.max_llm_calls,
                        current,
                        "model request limit reached",
                    )
            if scope.limits.max_tool_calls is not None:
                current = scope.tool_calls_used + scope.tool_calls_reserved
                if current >= scope.limits.max_tool_calls:
                    self._limit_error(
                        scope,
                        "max_tool_calls",
                        scope.limits.max_tool_calls,
                        current,
                        "tool execution limit reached",
                    )

    def _check_request_capacity(self, llm_calls: int, tool_calls: int) -> None:
        found = find_request_violation(self._scopes(), llm_calls, tool_calls)
        if found is not None:
            scope, limit_name, limit, current = found
            self._limit_error(
                scope, limit_name, limit, current, f"declared calls would exceed {limit_name}"
            )

    def _check_operation_cap(self, cost: Decimal) -> None:
        for scope in self._scopes():
            cap = scope.limits.max_operation_cost
            if cap is not None and cost > cap:
                self._guard._record_event(
                    scope=scope.name,
                    kind="operation_cap_rejected",
                    detail=f"operation estimate {cost} exceeds max_operation_cost {cap}",
                )
                raise BudgetExceededError(
                    f"{scope.name}: operation estimate {cost} exceeds max_operation_cost {cap}",
                    scope=scope.name,
                    limit=cap,
                    requested=cost,
                    available=cap,
                )

    def _limit_error(
        self, scope: Scope, limit_name: str, limit: int, current: int, reason: str
    ) -> NoReturn:
        self._guard._record_event(
            scope=scope.name,
            kind="limit_rejected",
            detail=f"{reason}: {limit_name}={limit}, current={current}",
        )
        raise LimitExceededError(
            f"{scope.name}: {reason} ({limit_name}={limit}, current={current})",
            scope=scope.name,
            limit_name=limit_name,
            limit=limit,
            current=current,
        )

    def _reject_for_budget(
        self,
        scope: Scope,
        *,
        requested: Decimal,
        available: Decimal,
        reason: str,
    ) -> NoReturn:
        guard = self._guard
        policy = guard.policies.on_budget_exceeded
        guard._record_event(
            scope=scope.name,
            kind="budget_rejected",
            detail=f"{reason}; requested={requested}, available={available}, policy={policy}",
        )
        message = (
            f"Budget exceeded in {scope.name}: {reason}. "
            f"Requested {requested}, available {available}."
        )
        if policy == "pause":
            self._state = "paused"
            guard._record_event(
                scope=self._scope.name,
                kind="session_paused",
                detail="awaiting approval; new operations are rejected",
            )
            error = ApprovalRequiredError(
                message + " Session paused; approval is required.",
                scope=scope.name,
                limit=scope.limits.max_cost,
                requested=requested,
                available=available,
            )
            raise error
        if policy == "stop":
            self._state = "stopped"
            guard._record_event(
                scope=self._scope.name, kind="session_stopped", detail="new operations are rejected"
            )
            message += " Session stopped."
        raise BudgetExceededError(
            message,
            scope=scope.name,
            limit=scope.limits.max_cost,
            requested=requested,
            available=available,
        )

    def _move_request_to_used(self, *, llm_calls: int, tool_calls: int) -> None:
        for scope in self._scopes():
            if llm_calls:
                scope.llm_calls_reserved -= llm_calls
                scope.llm_calls_used += llm_calls
            if tool_calls:
                scope.tool_calls_reserved -= tool_calls
                scope.tool_calls_used += tool_calls

    def _add_tokens(self, input_tokens: int | None, output_tokens: int | None, cached: int) -> None:
        for scope in self._scopes():
            if input_tokens:
                scope.input_tokens += input_tokens
            if output_tokens:
                scope.output_tokens += output_tokens
            if cached:
                scope.cached_input_tokens += cached

    def _token_limits_configured(self) -> bool:
        return any(
            scope.limits.max_tokens is not None
            or scope.limits.max_input_tokens_per_call is not None
            or scope.limits.max_output_tokens_per_call is not None
            for scope in self._scopes()
        )

    def _enforce_token_limits(self, input_tokens: int | None, output_tokens: int | None) -> None:
        for scope in self._scopes():
            limits = scope.limits
            if (
                input_tokens is not None
                and limits.max_input_tokens_per_call is not None
                and input_tokens > limits.max_input_tokens_per_call
            ):
                self._limit_error(
                    scope,
                    "max_input_tokens_per_call",
                    limits.max_input_tokens_per_call,
                    input_tokens,
                    "input tokens for one call exceeded the limit",
                )
            if (
                output_tokens is not None
                and limits.max_output_tokens_per_call is not None
                and output_tokens > limits.max_output_tokens_per_call
            ):
                self._limit_error(
                    scope,
                    "max_output_tokens_per_call",
                    limits.max_output_tokens_per_call,
                    output_tokens,
                    "output tokens for one call exceeded the limit",
                )
            if limits.max_tokens is not None and scope.total_tokens > limits.max_tokens:
                self._limit_error(
                    scope,
                    "max_tokens",
                    limits.max_tokens,
                    scope.total_tokens,
                    "total token limit exceeded",
                )
