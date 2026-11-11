"""Command-line interface tests, run in-process with Typer's CliRunner."""

import json

import pytest
from typer.testing import CliRunner

from owl_agent_budget_guard import __version__
from owl_agent_budget_guard.cli import app
from owl_agent_budget_guard.config import STARTER_CONFIG

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "owl-budget.yaml"
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    return path


def test_version_matches_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"owl-budget {__version__}"


def test_init_creates_starter_config(tmp_path) -> None:
    target = tmp_path / "new.yaml"
    result = runner.invoke(app, ["init", "--path", str(target)])
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8") == STARTER_CONFIG
    assert runner.invoke(app, ["validate", "--config", str(target)]).exit_code == 0


def test_init_does_not_overwrite_without_confirmation(config_file) -> None:
    config_file.write_text("keep me\n", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(config_file)], input="n\n")
    assert result.exit_code == 1
    assert "not changed" in result.output
    assert config_file.read_text(encoding="utf-8") == "keep me\n"


def test_init_with_confirmation_overwrites(config_file) -> None:
    config_file.write_text("old\n", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(config_file)], input="y\n")
    assert result.exit_code == 0
    assert config_file.read_text(encoding="utf-8") == STARTER_CONFIG


def test_init_without_input_does_not_overwrite(config_file) -> None:
    config_file.write_text("old\n", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(config_file)])
    assert result.exit_code != 0
    assert config_file.read_text(encoding="utf-8") == "old\n"


def test_init_force_overwrites_silently(config_file) -> None:
    config_file.write_text("old\n", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(config_file), "--force"])
    assert result.exit_code == 0
    assert config_file.read_text(encoding="utf-8") == STARTER_CONFIG


def test_validate_accepts_valid_file(config_file) -> None:
    result = runner.invoke(app, ["validate", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "OK:" in result.output
    assert "agent=my-agent" in result.output


def test_validate_reports_errors_with_exit_code_1(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nagent:\n  name: a\nbudget:\n  max_cost: -3\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--config", str(bad)])
    assert result.exit_code == 1
    assert "budget.max_cost" in result.output


def test_validate_missing_file_fails_cleanly(tmp_path) -> None:
    result = runner.invoke(app, ["validate", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_status_shows_limits_and_states_persistence_limit(config_file) -> None:
    result = runner.invoke(app, ["status", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "Budget (agent):     5.00000000 USD" in result.output
    assert "On budget exceeded: stop" in result.output
    assert "in memory" in result.output


def test_report_without_input_renders_empty_report_text(config_file) -> None:
    result = runner.invoke(app, ["report", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "OWL Agent Budget Guard report" in result.output
    assert "No usage was recorded" in result.output


def test_report_json_output_is_valid_json(config_file) -> None:
    result = runner.invoke(app, ["report", "--config", str(config_file), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["report_version"] == 1
    assert data["agent"]["budget"]["limit"] == "5.00000000"


def test_report_from_saved_file_with_valid_config(tmp_path, config_file, make_guard) -> None:
    guard = make_guard()
    with guard.session("saved") as session:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(model="test-model", input_tokens=1000, output_tokens=500)
        saved = session.report()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    result = runner.invoke(app, ["report", "--config", str(config_file), "--input", str(path)])
    assert result.exit_code == 0, result.output
    assert "0.0035 USD" not in result.output  # the text report shows the raw decimal strings
    assert "Actual spending:       0.00350000 USD" in result.output
    assert "Operations:            1 succeeded, 0 failed" in result.output


def test_report_rejects_files_that_are_not_reports(tmp_path, config_file) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    result = runner.invoke(app, ["report", "--config", str(config_file), "--input", str(path)])
    assert result.exit_code == 1
    assert "not an OWL budget report" in result.output


def test_report_rejects_invalid_json(tmp_path, config_file) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["report", "--config", str(config_file), "--input", str(path)])
    assert result.exit_code == 1
    assert "not valid JSON" in result.output


def test_report_rejects_unknown_format(config_file) -> None:
    result = runner.invoke(app, ["report", "--config", str(config_file), "--format", "xml"])
    assert result.exit_code == 2


def test_pricing_lists_configured_models(tmp_path) -> None:
    path = tmp_path / "priced.yaml"
    path.write_text(
        "version: 1\nagent:\n  name: a\nbudget:\n  max_cost: 1\n"
        "pricing:\n  models:\n    alpha-model:\n      input_per_million: 1\n"
        "      output_per_million: 2.5\n      source: test\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["pricing", "--config", str(path)])
    assert result.exit_code == 0
    assert "alpha-model" in result.output
    assert "2.5" in result.output
    as_json = runner.invoke(app, ["pricing", "--config", str(path), "--format", "json"])
    assert json.loads(as_json.output)["models"][0]["model"] == "alpha-model"


def test_pricing_with_no_models_says_no_prices_are_bundled(config_file) -> None:
    result = runner.invoke(app, ["pricing", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "No prices are bundled" in result.output


def test_unknown_command_is_rejected() -> None:
    result = runner.invoke(app, ["frobnicate"])
    assert result.exit_code == 2
