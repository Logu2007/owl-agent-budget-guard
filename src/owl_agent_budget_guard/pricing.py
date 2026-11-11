"""Model pricing registry and local cost calculation.

No prices are bundled with the package. Register the prices you have verified
from your provider, either in code with :meth:`PricingRegistry.register` or in
the ``pricing.models`` section of the configuration file.

Costs computed here are *locally calculated estimates of provider billing*.
They are not provider invoices. Use ``reported_cost`` when the provider returns
the actual cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext

from owl_agent_budget_guard.exceptions import InvalidValueError, PricingError
from owl_agent_budget_guard.money import MONEY_QUANTUM, ZERO, to_money
from owl_agent_budget_guard.validation import require_count, validate_currency, validate_name

_TOKENS_PER_UNIT = Decimal(1_000_000)
_MAX_SOURCE_LENGTH = 500


@dataclass(frozen=True)
class ModelPricing:
    """Prices for one model, expressed per one million tokens.

    Attributes:
        model: Model identifier as used in ``record_llm_call(model=...)``.
        input_per_million: Price of one million uncached input tokens.
        output_per_million: Price of one million output tokens.
        currency: Three-letter currency code. Must match the guard currency.
        cached_input_per_million: Price of one million cached input tokens.
            ``None`` means cached tokens cannot be priced and recording them
            raises :class:`PricingError`.
        source: Free-text note on where the prices came from and when they were
            verified, for example a URL and a date.
    """

    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    currency: str = "USD"
    cached_input_per_million: Decimal | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        validate_name(self.model, "model")
        validate_currency(self.currency)
        object.__setattr__(
            self, "input_per_million", to_money(self.input_per_million, "input_per_million")
        )
        object.__setattr__(
            self, "output_per_million", to_money(self.output_per_million, "output_per_million")
        )
        if self.cached_input_per_million is not None:
            object.__setattr__(
                self,
                "cached_input_per_million",
                to_money(self.cached_input_per_million, "cached_input_per_million"),
            )
        if self.source is not None and (
            not isinstance(self.source, str) or len(self.source) > _MAX_SOURCE_LENGTH
        ):
            raise InvalidValueError(
                f"source must be a string of at most {_MAX_SOURCE_LENGTH} characters"
            )


class PricingRegistry:
    """In-process collection of :class:`ModelPricing` entries."""

    def __init__(self) -> None:
        self._models: dict[str, ModelPricing] = {}

    def register(self, pricing: ModelPricing, *, replace: bool = False) -> None:
        """Add pricing for a model. Refuses to overwrite unless ``replace=True``."""
        if not isinstance(pricing, ModelPricing):
            raise PricingError("register() expects a ModelPricing instance")
        if pricing.model in self._models and not replace:
            raise PricingError(
                f"pricing for model {pricing.model!r} is already registered; "
                "pass replace=True to overwrite it"
            )
        self._models[pricing.model] = pricing

    def get(self, model: str) -> ModelPricing:
        """Return pricing for ``model`` or raise :class:`PricingError`."""
        try:
            return self._models[model]
        except KeyError as exc:
            raise PricingError(f"no pricing registered for model {model!r}") from exc

    def __contains__(self, model: object) -> bool:
        return model in self._models

    def all(self) -> list[ModelPricing]:
        """Return all registered pricing entries sorted by model name."""
        return [self._models[name] for name in sorted(self._models)]

    def calculate_cost(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        currency: str,
        cached_input_tokens: int = 0,
    ) -> Decimal:
        """Return the locally calculated cost of one model call.

        ``cached_input_tokens`` are a subset of ``input_tokens`` and are billed at
        the cached rate. The result is rounded *up* to eight decimal places so
        that rounding never under-reports cost against the budget.
        """
        pricing = self.get(model)
        if pricing.currency != currency:
            raise PricingError(
                f"pricing for model {model!r} is in {pricing.currency}, "
                f"but the budget currency is {currency}"
            )
        input_count = require_count(input_tokens, "input_tokens")
        output_count = require_count(output_tokens, "output_tokens")
        cached_count = require_count(cached_input_tokens, "cached_input_tokens")
        if cached_count > input_count:
            raise InvalidValueError("cached_input_tokens must not exceed input_tokens")
        if cached_count and pricing.cached_input_per_million is None:
            raise PricingError(f"model {model!r} has no cached_input_per_million price")

        cached_rate = pricing.cached_input_per_million or ZERO
        uncached_count = input_count - cached_count
        with localcontext() as context:
            context.prec = 60
            total = (
                Decimal(uncached_count) * pricing.input_per_million
                + Decimal(cached_count) * cached_rate
                + Decimal(output_count) * pricing.output_per_million
            ) / _TOKENS_PER_UNIT
            return total.quantize(MONEY_QUANTUM, rounding=ROUND_CEILING)
