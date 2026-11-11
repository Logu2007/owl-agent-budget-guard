"""YAML configuration: safe loading, strict validation, and conversion to SDK objects.

Security properties:

* Only ``yaml.SafeLoader`` is used, so no Python objects are constructed.
* Duplicate keys are rejected instead of silently keeping the last value.
* Unknown keys are rejected at every level.
* Files larger than :data:`MAX_CONFIG_BYTES` are rejected before parsing.
* Error messages report locations and rule names, never the file's values.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError

from owl_agent_budget_guard.exceptions import ConfigError
from owl_agent_budget_guard.guard import BudgetGuard
from owl_agent_budget_guard.limits import Limits, Policies
from owl_agent_budget_guard.money import to_money
from owl_agent_budget_guard.pricing import ModelPricing
from owl_agent_budget_guard.store import SqliteStore
from owl_agent_budget_guard.validation import CURRENCY_PATTERN_TEXT, NAME_PATTERN_TEXT

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024


def _money_input(value: object) -> Decimal:
    return to_money(value, "amount")


def _no_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be a number, not a boolean")
    return value


Money = Annotated[Decimal, BeforeValidator(_money_input)]
NonNegativeInt = Annotated[int, BeforeValidator(_no_bool), Field(ge=0, strict=True)]
PositiveSeconds = Annotated[float, BeforeValidator(_no_bool), Field(gt=0, allow_inf_nan=False)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentSection(_Section):
    name: str = Field(pattern=f"^{NAME_PATTERN_TEXT}$")


class BudgetSection(_Section):
    max_cost: Money
    currency: str = Field(default="USD", pattern=f"^{CURRENCY_PATTERN_TEXT}$")


class LimitsSection(_Section):
    max_cost_per_operation: Money | None = None
    max_llm_calls: NonNegativeInt | None = None
    max_tool_calls: NonNegativeInt | None = None
    max_tokens: NonNegativeInt | None = None
    max_input_tokens_per_call: NonNegativeInt | None = None
    max_output_tokens_per_call: NonNegativeInt | None = None
    max_execution_time: PositiveSeconds | None = None


class PoliciesSection(_Section):
    on_budget_exceeded: Literal["stop", "raise", "pause"] = "stop"
    unknown_cost: Literal["block", "estimate"] = "block"
    allow_budget_extension: bool = False

    @field_validator("allow_budget_extension")
    @classmethod
    def _extension_not_supported(cls, value: bool) -> bool:
        if value:
            raise ValueError("automatic budget extension is not supported; keep this set to false")
        return value


class PricingModelSection(_Section):
    currency: str = Field(default="USD", pattern=f"^{CURRENCY_PATTERN_TEXT}$")
    input_per_million: Money
    output_per_million: Money
    cached_input_per_million: Money | None = None
    source: str | None = Field(default=None, max_length=500)


class PricingSection(_Section):
    models: dict[str, PricingModelSection] = Field(default_factory=dict)


class StoreSection(_Section):
    """Optional SQLite file for persistent budget state.

    A relative path is resolved against the current working directory.
    """

    path: str = Field(min_length=1, max_length=1024)


class GuardConfig(_Section):
    """The complete, validated configuration file."""

    version: Literal[1]
    agent: AgentSection
    budget: BudgetSection
    limits: LimitsSection = Field(default_factory=LimitsSection)
    policies: PoliciesSection = Field(default_factory=PoliciesSection)
    pricing: PricingSection = Field(default_factory=PricingSection)
    store: StoreSection | None = None

    @model_validator(mode="after")
    def _pricing_matches_budget_currency(self) -> GuardConfig:
        for model_name, model_pricing in self.pricing.models.items():
            if model_pricing.currency != self.budget.currency:
                raise ValueError(
                    f"pricing for model {model_name!r} is in {model_pricing.currency}, "
                    f"but budget.currency is {self.budget.currency}"
                )
        return self


STARTER_CONFIG = """\
# OWL Agent Budget Guard configuration.
# See the README ("Configuration") for every field.
version: 1

agent:
  name: my-agent

budget:
  max_cost: 5.00          # agent-wide spending cap, in `currency`
  currency: USD

limits:
  max_llm_calls: 50
  max_tool_calls: 100
  max_tokens: 100000
  max_input_tokens_per_call: 20000
  max_output_tokens_per_call: 10000
  max_execution_time: 600  # seconds, measured with a monotonic clock
  # max_cost_per_operation: 0.50

policies:
  on_budget_exceeded: stop  # stop | raise | pause
  unknown_cost: block       # block | estimate
  allow_budget_extension: false

pricing:
  models: {}
  # No prices are bundled. Add the prices you have verified, for example:
  # example-model:
  #   currency: USD
  #   input_per_million: 1.00
  #   output_per_million: 5.00
  #   source: "https://example.com/pricing (verified 2026-01-01)"
"""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in seen
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _format_yaml_error(source: str, exc: yaml.YAMLError) -> str:
    # Do not include str(exc): it may quote file contents.
    problem = getattr(exc, "problem", None) or "invalid YAML"
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is not None:
        return f"{source}: line {mark.line + 1}, column {mark.column + 1}: {problem}"
    return f"{source}: {problem}"


def _format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"{source}: configuration is invalid"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def parse_config_text(text: str, source: str = "<config>") -> GuardConfig:
    """Parse and validate configuration text. Raises :class:`ConfigError`."""
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ConfigError(f"{source}: file is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(_format_yaml_error(source, exc)) from None
    if data is None:
        raise ConfigError(f"{source}: configuration file is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: the top level must be a mapping of sections")
    try:
        return GuardConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(source, exc)) from None


def load_config(path: str | Path) -> GuardConfig:
    """Read, parse and validate a configuration file. Raises :class:`ConfigError`."""
    config_path = Path(path)
    try:
        size = config_path.stat().st_size
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {config_path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {config_path}: {exc.strerror}") from None
    if size > MAX_CONFIG_BYTES:
        raise ConfigError(f"{config_path}: file is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        text = config_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ConfigError(f"{config_path}: file is not valid UTF-8") from None
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {config_path}: {exc.strerror}") from None
    return parse_config_text(text, source=str(config_path))


def build_limits(config: GuardConfig) -> Limits:
    section = config.limits
    return Limits(
        max_cost=config.budget.max_cost,
        max_operation_cost=section.max_cost_per_operation,
        max_llm_calls=section.max_llm_calls,
        max_tool_calls=section.max_tool_calls,
        max_tokens=section.max_tokens,
        max_input_tokens_per_call=section.max_input_tokens_per_call,
        max_output_tokens_per_call=section.max_output_tokens_per_call,
        max_execution_time=section.max_execution_time,
    )


def build_pricing(config: GuardConfig) -> list[ModelPricing]:
    return [
        ModelPricing(
            model=name,
            input_per_million=entry.input_per_million,
            output_per_million=entry.output_per_million,
            currency=entry.currency,
            cached_input_per_million=entry.cached_input_per_million,
            source=entry.source,
        )
        for name, entry in config.pricing.models.items()
    ]


def build_guard(config: GuardConfig) -> BudgetGuard:
    """Create a :class:`BudgetGuard` from validated configuration."""
    return BudgetGuard(
        agent_name=config.agent.name,
        currency=config.budget.currency,
        limits=build_limits(config),
        policies=Policies(
            on_budget_exceeded=config.policies.on_budget_exceeded,
            unknown_cost=config.policies.unknown_cost,
        ),
        pricing=build_pricing(config),
        store=None if config.store is None else SqliteStore(config.store.path),
    )
