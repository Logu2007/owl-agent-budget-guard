"""OWL Agent Budget Guard: local budget, token, request, tool and time limits for AI agent runs.

The guard measures and enforces only what the caller reports to it. It does not
intercept provider API calls. See the README for the enforcement boundaries.
"""

from owl_agent_budget_guard.exceptions import (
    ApprovalRequiredError,
    BudgetExceededError,
    ConfigError,
    ExecutionTimeExceededError,
    InvalidValueError,
    LimitExceededError,
    OwlBudgetError,
    PricingError,
    ReservationError,
    ReservationOverrunError,
    SessionClosedError,
    SessionStoppedError,
    UnknownCostError,
)
from owl_agent_budget_guard.guard import BudgetGuard
from owl_agent_budget_guard.limits import Limits, Policies
from owl_agent_budget_guard.pricing import ModelPricing, PricingRegistry
from owl_agent_budget_guard.reservation import Reservation
from owl_agent_budget_guard.session import Session

__version__ = "0.1.0"

__all__ = [
    "ApprovalRequiredError",
    "BudgetExceededError",
    "BudgetGuard",
    "ConfigError",
    "ExecutionTimeExceededError",
    "InvalidValueError",
    "LimitExceededError",
    "Limits",
    "ModelPricing",
    "OwlBudgetError",
    "Policies",
    "PricingError",
    "PricingRegistry",
    "Reservation",
    "ReservationError",
    "ReservationOverrunError",
    "Session",
    "SessionClosedError",
    "SessionStoppedError",
    "UnknownCostError",
    "__version__",
]
