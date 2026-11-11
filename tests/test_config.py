"""Configuration: safe YAML loading, validation messages, and conversion to SDK objects."""

from decimal import Decimal

import pytest

from owl_agent_budget_guard import BudgetExceededError, ConfigError
from owl_agent_budget_guard.config import (
    MAX_CONFIG_BYTES,
    STARTER_CONFIG,
    build_guard,
    load_config,
    parse_config_text,
)

VALID = """
version: 1
agent:
  name: owl-security-agent
budget:
  max_cost: 5.00
  currency: USD
limits:
  max_llm_calls: 50
  max_tokens: 100000
  max_input_tokens_per_call: 20000
  max_output_tokens_per_call: 10000
  max_tool_calls: 100
  max_execution_time: 600
policies:
  on_budget_exceeded: stop
  unknown_cost: block
  allow_budget_extension: false
pricing:
  models:
    example-model:
      currency: USD
      input_per_million: 1.00
      output_per_million: 5.00
      source: "test fixture"
"""


def _with_budget(extra_lines: str) -> str:
    return f"version: 1\nagent:\n  name: a\nbudget:\n  max_cost: 1\n{extra_lines}"


def test_valid_configuration_loads_with_types_and_defaults() -> None:
    config = parse_config_text(VALID)
    assert config.agent.name == "owl-security-agent"
    assert config.budget.max_cost == Decimal("5.00")
    assert config.limits.max_execution_time == 600.0
    assert config.policies.on_budget_exceeded == "stop"
    assert config.pricing.models["example-model"].input_per_million == Decimal("1.00000000")


def test_minimal_configuration_uses_safe_defaults() -> None:
    config = parse_config_text(_with_budget(""))
    assert config.budget.currency == "USD"
    assert config.policies.on_budget_exceeded == "stop"
    assert config.policies.unknown_cost == "block"
    assert config.limits.max_llm_calls is None  # unlimited unless configured


def test_starter_configuration_is_valid() -> None:
    config = parse_config_text(STARTER_CONFIG, source="starter")
    assert config.budget.max_cost == Decimal("5.00000000")


def test_float_costs_are_exact() -> None:
    config = parse_config_text(_with_budget("").replace("max_cost: 1", "max_cost: 0.1"))
    assert config.budget.max_cost == Decimal("0.10000000")


@pytest.mark.parametrize(
    ("text", "expected_fragment"),
    [
        ("version: 1\nagent: [unclosed\n", "line"),
        (
            "version: 1\nversion: 1\nagent:\n  name: a\nbudget:\n  max_cost: 1\n",
            "duplicate key 'version'",
        ),
        ("- just\n- a list\n", "top level must be a mapping"),
        ("", "empty"),
    ],
)
def test_malformed_yaml_and_structure_are_rejected(text: str, expected_fragment: str) -> None:
    with pytest.raises(ConfigError, match=expected_fragment.replace("[", r"\[")):
        parse_config_text(text, source="test.yaml")


def test_yaml_error_does_not_echo_file_contents() -> None:
    secret = "sk-test-SECRET-VALUE-123"
    text = f"version: 1\napi_key: {secret}\nagent: [unclosed\n"
    with pytest.raises(ConfigError) as info:
        parse_config_text(text)
    assert secret not in str(info.value)


def test_unsafe_yaml_tags_are_not_constructed() -> None:
    text = "version: !!python/object/apply:os.system ['echo hacked']\n"
    with pytest.raises(ConfigError):
        parse_config_text(text)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (_with_budget("  max_cost_typo: 1\n"), "max_cost_typo"),
        (_with_budget("unknown_section:\n  a: 1\n"), "unknown_section"),
        (_with_budget("limits:\n  max_llm_calls: 5\n  max_llm_call: 6\n"), "limits.max_llm_call"),
        (_with_budget("policies:\n  on_budget_exceeded: retry\n"), "policies.on_budget_exceeded"),
        (_with_budget("policies:\n  unknown_cost: assume_zero\n"), "policies.unknown_cost"),
    ],
)
def test_unknown_fields_and_invalid_choices_are_rejected(text: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment.replace(".", r"\.")):
        parse_config_text(text)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("version: 1\nagent:\n  name: a\nbudget:\n  max_cost: -1\n", "budget.max_cost"),
        (_with_budget("limits:\n  max_llm_calls: -5\n"), "limits.max_llm_calls"),
        (_with_budget("limits:\n  max_tokens: -1\n"), "limits.max_tokens"),
        (_with_budget("limits:\n  max_execution_time: 0\n"), "limits.max_execution_time"),
        (_with_budget("limits:\n  max_execution_time: -10\n"), "limits.max_execution_time"),
    ],
)
def test_negative_or_zero_limits_are_rejected(text: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment.replace(".", r"\.")):
        parse_config_text(text)


def test_incorrect_types_are_rejected() -> None:
    with pytest.raises(ConfigError, match="budget.max_cost"):
        parse_config_text("version: 1\nagent:\n  name: a\nbudget:\n  max_cost: abc\n")
    with pytest.raises(ConfigError, match="limits.max_llm_calls"):
        parse_config_text(_with_budget("limits:\n  max_llm_calls: 1.5\n"))
    with pytest.raises(ConfigError, match="limits.max_llm_calls"):
        parse_config_text(_with_budget("limits:\n  max_llm_calls: true\n"))
    with pytest.raises(ConfigError, match="budget.max_cost"):
        parse_config_text("version: 1\nagent:\n  name: a\nbudget:\n  max_cost: true\n")


def test_missing_required_values_are_reported() -> None:
    with pytest.raises(ConfigError) as info:
        parse_config_text("version: 1\nagent:\n  name: a\n")
    assert "budget" in str(info.value)
    with pytest.raises(ConfigError, match="version"):
        parse_config_text("agent:\n  name: a\nbudget:\n  max_cost: 1\n")
    with pytest.raises(ConfigError, match="version"):
        parse_config_text("version: 2\nagent:\n  name: a\nbudget:\n  max_cost: 1\n")


def test_budget_extension_cannot_be_enabled() -> None:
    with pytest.raises(ConfigError, match="automatic budget extension is not supported"):
        parse_config_text(_with_budget("policies:\n  allow_budget_extension: true\n"))


def test_pricing_currency_must_match_budget_currency() -> None:
    text = (
        "version: 1\nagent:\n  name: a\nbudget:\n  max_cost: 1\n  currency: EUR\n"
        "pricing:\n  models:\n    m:\n      currency: USD\n"
        "      input_per_million: 1\n      output_per_million: 1\n"
    )
    with pytest.raises(ConfigError, match="budget.currency is EUR"):
        parse_config_text(text)


def test_invalid_agent_name_is_rejected() -> None:
    with pytest.raises(ConfigError, match="agent.name"):
        parse_config_text("version: 1\nagent:\n  name: '   '\nbudget:\n  max_cost: 1\n")


def test_file_loading_errors(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    big = tmp_path / "big.yaml"
    big.write_text("#" * (MAX_CONFIG_BYTES + 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="larger than"):
        load_config(big)
    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config(binary)
    good = tmp_path / "good.yaml"
    good.write_text(VALID, encoding="utf-8")
    assert load_config(good).agent.name == "owl-security-agent"


def test_duplicate_keys_are_rejected_in_nested_sections() -> None:
    text = _with_budget("limits:\n  max_llm_calls: 5\n  max_llm_calls: 9\n")
    with pytest.raises(ConfigError, match="duplicate key 'max_llm_calls'"):
        parse_config_text(text)


def test_built_guard_enforces_configured_limits() -> None:
    config = parse_config_text(VALID)
    guard = build_guard(config)
    assert guard.agent_name == "owl-security-agent"
    assert guard.limits.max_cost == Decimal("5.00000000")
    assert guard.limits.max_llm_calls == 50
    assert "example-model" in guard.pricing
    session = guard.session("from-config")
    with pytest.raises(BudgetExceededError):
        session.reserve("6")
    session.close()
