"""Defensive branches: constructor type checks, file-system errors, and unusual inputs."""

from decimal import Decimal

import pytest
from typer.testing import CliRunner

from owl_agent_budget_guard import (
    BudgetGuard,
    ConfigError,
    InvalidValueError,
    LimitExceededError,
    Limits,
    ModelPricing,
    Policies,
    PricingError,
)
from owl_agent_budget_guard.cli import app
from owl_agent_budget_guard.config import load_config, parse_config_text
from owl_agent_budget_guard.session import Session

runner = CliRunner()


def test_tool_call_beyond_declared_count_is_refused(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.10", tool_calls=1)
        reservation.record_tool_call(cost=0)
        with pytest.raises(LimitExceededError, match="declared 1 tool call"):
            reservation.record_tool_call(cost=0)
        reservation.release()


def test_guard_rejects_wrong_argument_types() -> None:
    with pytest.raises(TypeError, match="limits"):
        BudgetGuard(limits={"max_cost": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policies"):
        BudgetGuard(policies="stop")  # type: ignore[arg-type]


def test_guard_rejects_invalid_agent_name_and_currency() -> None:
    with pytest.raises(InvalidValueError, match="agent_name"):
        BudgetGuard(agent_name="")
    with pytest.raises(InvalidValueError, match="currency"):
        BudgetGuard(currency="usd")


def test_session_rejects_non_limits_object() -> None:
    guard = BudgetGuard()
    with pytest.raises(TypeError, match="Limits instance"):
        Session(guard=guard, name="x", limits={"max_cost": 1})  # type: ignore[arg-type]


def test_session_name_is_validated(make_guard) -> None:
    guard = make_guard()
    with pytest.raises(InvalidValueError, match="session name"):
        guard.session("bad\nname")


def test_register_pricing_rejects_non_pricing_objects(make_guard) -> None:
    guard = make_guard(pricing=False)
    with pytest.raises(PricingError, match="ModelPricing instance"):
        guard.register_pricing({"model": "x"})  # type: ignore[arg-type]


def test_pricing_registry_rejects_non_pricing_objects() -> None:
    from owl_agent_budget_guard import PricingRegistry

    with pytest.raises(PricingError, match="ModelPricing instance"):
        PricingRegistry().register("nope")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_execution_time": True},
        {"max_execution_time": float("inf")},
        {"max_execution_time": "10"},
        {"max_llm_calls": True},
        {"max_tokens": -1},
        {"max_cost": "abc"},
    ],
)
def test_limits_reject_invalid_values(kwargs: dict) -> None:
    with pytest.raises(InvalidValueError):
        Limits(**kwargs)


def test_policies_reject_unknown_values() -> None:
    with pytest.raises(InvalidValueError, match="on_budget_exceeded"):
        Policies(on_budget_exceeded="retry")  # type: ignore[arg-type]
    with pytest.raises(InvalidValueError, match="unknown_cost"):
        Policies(unknown_cost="free")  # type: ignore[arg-type]


def test_model_pricing_positional_defaults() -> None:
    pricing = ModelPricing("m", Decimal("1"), Decimal("2"))
    assert pricing.currency == "USD"
    assert pricing.cached_input_per_million is None


def test_config_rejects_unhashable_mapping_key() -> None:
    with pytest.raises(ConfigError):
        parse_config_text("? [a, b]\n: 1\nversion: 1\n")


def test_yaml_error_without_mark_still_reports_source() -> None:
    import yaml

    from owl_agent_budget_guard.config import _format_yaml_error

    assert _format_yaml_error("src.yaml", yaml.YAMLError()) == "src.yaml: invalid YAML"


def test_load_config_on_a_directory_reports_read_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="cannot read configuration file"):
        load_config(tmp_path)


def test_load_config_on_non_utf8_is_reported(tmp_path) -> None:
    path = tmp_path / "latin.yaml"
    path.write_bytes("name: caf\xe9\n".encode("latin-1"))
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config(path)


def test_init_into_a_directory_reports_write_error(tmp_path) -> None:
    result = runner.invoke(app, ["init", "--path", str(tmp_path), "--force"])
    assert result.exit_code == 1
    assert "cannot write" in result.output


def test_report_with_non_utf8_file_is_reported(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe")
    from owl_agent_budget_guard.config import STARTER_CONFIG

    config = tmp_path / "owl-budget.yaml"
    config.write_text(STARTER_CONFIG, encoding="utf-8")
    result = runner.invoke(app, ["report", "--config", str(config), "--input", str(path)])
    assert result.exit_code == 1
    assert "cannot read" in result.output


def test_report_and_pricing_reject_unknown_format(tmp_path) -> None:
    from owl_agent_budget_guard.config import STARTER_CONFIG

    config = tmp_path / "owl-budget.yaml"
    config.write_text(STARTER_CONFIG, encoding="utf-8")
    assert (
        runner.invoke(app, ["pricing", "--config", str(config), "--format", "xml"]).exit_code == 2
    )


def test_report_json_from_saved_file_round_trips(tmp_path, make_guard) -> None:
    import json

    from owl_agent_budget_guard.config import STARTER_CONFIG

    config = tmp_path / "owl-budget.yaml"
    config.write_text(STARTER_CONFIG, encoding="utf-8")
    guard = make_guard()
    with guard.session("saved") as session:
        saved = session.report()
    saved_path = tmp_path / "saved.json"
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    result = runner.invoke(
        app, ["report", "--config", str(config), "--input", str(saved_path), "--format", "json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["session"]["name"] == "saved"
