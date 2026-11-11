"""Configured stores: config section, read-only CLI views, and store-backed build_guard."""

import json
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from owl_agent_budget_guard import Limits
from owl_agent_budget_guard.cli import app
from owl_agent_budget_guard.config import build_guard, parse_config_text
from owl_agent_budget_guard.exceptions import ConfigError, PersistenceError
from owl_agent_budget_guard.store import SqliteStore
from owl_agent_budget_guard.store_report import build_store_report, format_store_report_text

runner = CliRunner()


def _config_text(store_path, agent="store-agent") -> str:
    return (
        f"version: 1\nagent:\n  name: {agent}\nbudget:\n  max_cost: 5\n  currency: USD\n"
        f"store:\n  path: {store_path}\n"
    )


@pytest.fixture
def stored(tmp_path):
    """A store with one settled session, one recorded event, and a live guard that owns it."""
    db = tmp_path / "budget.db"
    guard = build_guard(parse_config_text(_config_text(db)))
    session = guard.session("nightly")
    with session.operation("0.20", llm_calls=1) as op:
        op.record_llm_call(reported_cost="0.07", input_tokens=10, output_tokens=5)
    return db, guard, session


def test_config_store_section_is_optional_and_validated(tmp_path) -> None:
    assert parse_config_text(_config_text(tmp_path / "x.db")).store.path.endswith("x.db")
    with pytest.raises(ConfigError, match="store.path"):
        parse_config_text(
            "version: 1\nagent:\n  name: a\nbudget:\n  max_cost: 1\nstore:\n  path: ''\n"
        )
    with pytest.raises(ConfigError, match="store.unknown"):
        parse_config_text(
            "version: 1\nagent:\n  name: a\nbudget:\n  max_cost: 1\n"
            "store:\n  path: x.db\n  unknown: 1\n"
        )


def test_build_guard_opens_the_configured_store(tmp_path) -> None:
    db = tmp_path / "configured.db"
    guard = build_guard(parse_config_text(_config_text(db)))
    assert guard.store is not None
    assert guard.store.path == db
    guard.close()


def test_status_shows_stored_state_without_claiming_store(stored) -> None:
    db, guard, session = stored
    result = runner.invoke(app, ["status", "--config", _write_config(db)])
    assert result.exit_code == 0, result.output
    assert "Actual spending:    0.07000000 USD" in result.output
    assert "Sessions stored:    1" in result.output
    # The live guard still owns the store and can keep writing.
    session.reserve("0.01").release()
    guard.close()


def test_report_from_store_is_read_only_and_renders_text(stored) -> None:
    db, guard, _ = stored
    result = runner.invoke(app, ["report", "--config", _write_config(db)])
    assert result.exit_code == 0, result.output
    assert "OWL Agent Budget Guard: stored budget" in result.output
    assert "nightly" in result.output
    assert "orphaned" not in result.output
    guard.close()


def test_report_from_store_as_json(stored) -> None:
    db, guard, _ = stored
    result = runner.invoke(app, ["report", "--config", _write_config(db), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["source"] == "store"
    assert data["agent"]["budget"]["actual_spending"] == "0.07000000"
    assert data["sessions"][0]["name"] == "nightly"
    guard.close()


def test_store_report_rejects_a_different_agent(stored) -> None:
    db, guard, _ = stored
    with pytest.raises(PersistenceError, match="belongs to agent 'store-agent'"):
        build_store_report(db, agent_name="other", currency="USD", limits=Limits())
    guard.close()


def test_store_report_rejects_empty_store(tmp_path) -> None:
    empty = tmp_path / "empty.db"
    SqliteStore(empty).close()
    with pytest.raises(PersistenceError, match="has no recorded agent"):
        build_store_report(empty, agent_name="a", currency="USD", limits=Limits())


def test_status_reports_missing_store_cleanly(tmp_path) -> None:
    result = runner.invoke(app, ["status", "--config", _write_config(tmp_path / "absent.db")])
    assert result.exit_code == 1
    assert "no budget store" in result.output


def test_text_report_lists_sessions_and_events(stored) -> None:
    db, guard, _ = stored
    report = build_store_report(
        db, agent_name="store-agent", currency="USD", limits=Limits(max_cost=Decimal("5"))
    )
    text = format_store_report_text(report)
    assert "Sessions (1)" in text
    assert "spent 0.07000000" in text
    guard.close()


def _write_config(db) -> str:
    """Write a configuration next to the store and return its path."""
    path = db.parent / "owl-budget.yaml"
    path.write_text(_config_text(db), encoding="utf-8")
    return str(path)
