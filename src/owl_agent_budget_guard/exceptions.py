"""Exception types raised by OWL Agent Budget Guard.

Every error raised by the package derives from :class:`OwlBudgetError`, so
applications can catch the whole family with one ``except`` clause.
"""

from __future__ import annotations

from decimal import Decimal


class OwlBudgetError(Exception):
    """Base class for all OWL Agent Budget Guard errors."""


class ConfigError(OwlBudgetError):
    """The configuration file is missing, malformed, or fails validation."""


class InvalidValueError(OwlBudgetError, ValueError):
    """An input value (money, token count, limit, name, ...) is invalid.

    This is a programming or configuration error, not a budget decision.
    """


class PricingError(OwlBudgetError):
    """Pricing is missing, invalid, or incompatible with the budget currency."""


class UnknownCostError(OwlBudgetError):
    """A cost or usage value could not be determined and the policy blocks it.

    Raised when ``unknown_cost`` is ``"block"`` and an operation records a
    model or tool call without a cost, or completes without recording any cost.
    """


class BudgetExceededError(OwlBudgetError):
    """A reservation or check was rejected because the budget is insufficient.

    Attributes are ``None`` when the information does not apply.
    """

    def __init__(
        self,
        message: str,
        *,
        scope: str | None = None,
        limit: Decimal | None = None,
        requested: Decimal | None = None,
        available: Decimal | None = None,
    ) -> None:
        super().__init__(message)
        self.scope = scope
        self.limit = limit
        self.requested = requested
        self.available = available


class ApprovalRequiredError(BudgetExceededError):
    """Raised under the ``pause`` policy; the session is paused.

    The SDK has no approval API in this release. A paused session rejects all
    new operations; the application must decide whether to create a new
    session with explicit limits.
    """

    state = "approval_required"


class LimitExceededError(OwlBudgetError):
    """A non-monetary limit (requests, tools, tokens) was reached or exceeded."""

    def __init__(
        self,
        message: str,
        *,
        scope: str,
        limit_name: str,
        limit: int | float,
        current: int | float,
    ) -> None:
        super().__init__(message)
        self.scope = scope
        self.limit_name = limit_name
        self.limit = limit
        self.current = current


class ExecutionTimeExceededError(LimitExceededError):
    """The configured execution duration was reached.

    This is detection only. The guard cannot interrupt Python code or a remote
    request that is already running; it refuses new work and reports the event.
    """


class ReservationOverrunError(OwlBudgetError):
    """The actual cost of an operation was higher than its reservation.

    The actual cost has been recorded before this error is raised.
    """


class ReservationError(OwlBudgetError):
    """A reservation was used in an invalid state (double settlement, use after close)."""


class PersistenceError(OwlBudgetError):
    """The budget store could not be opened, read, or written.

    After a failed write the guard refuses further work, because its in-memory
    state may no longer match the store. Reopen the guard to continue.
    """


class StoreOwnershipError(PersistenceError):
    """Another guard instance claimed the store, so this guard may no longer write."""


class SessionStoppedError(OwlBudgetError):
    """The session was stopped by the ``stop`` policy and rejects new operations."""


class SessionClosedError(OwlBudgetError):
    """The session has been closed and rejects new operations."""
