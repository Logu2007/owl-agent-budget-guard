"""Usage report structure, JSON compatibility, and estimate/actual separation."""

import json
from decimal import Decimal

from tests.conftest import TEST_MODEL

from owl_agent_budget_guard import Limits
from owl_agent_budget_guard.report import REPORT_VERSION, format_report_text


def test_report_has_consistent_top_level_structure(make_guard) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session("structure") as session:
        report = session.report()
    assert report["report_version"] == REPORT_VERSION
    assert report["currency"] == "USD"
    assert report["agent"]["name"] == "test-agent"
    assert report["session"]["name"] == "structure"
    assert report["session"]["id"].startswith("ses_")
    for section in ("budget", "usage", "execution"):
        assert section in report["session"]
        assert section in report["agent"]
    assert set(report["session"]["budget"]["cost_by_source"]) == {
        "reported",
        "calculated",
        "reconciled",
        "estimated_fallback",
    }


def test_report_is_json_serializable_and_money_is_string(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(model=TEST_MODEL, input_tokens=1000, output_tokens=500)
        report = session.report()
    encoded = json.loads(json.dumps(report))
    assert isinstance(encoded["agent"]["budget"]["actual_spending"], str)
    assert encoded["agent"]["budget"]["actual_spending"] == "0.00350000"


def test_estimates_and_actuals_are_reported_separately(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.40")
        report = session.report()
        reservation.release()
    budget = report["agent"]["budget"]
    assert budget["reserved_amount"] == "0.40000000"
    assert budget["actual_spending"] == "0.00000000"


def test_text_report_mentions_estimates_and_events(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.10")),
    )
    with guard.session(unique_name) as session:
        try:
            session.reserve("1")
        except Exception:
            pass
        text = format_report_text(session.report())
    assert "Reserved (estimate)" in text
    assert "budget_rejected" in text
    assert "Enforcement events" in text


def test_session_report_includes_only_its_own_and_agent_events(make_guard) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    first = guard.session("first", limits=Limits(max_cost=Decimal("0.01")))
    second = guard.session("second")
    try:
        first.reserve("0.50")
    except Exception:
        pass
    report_second = second.report()
    assert all(
        event["scope"] != f"session:{first.session_id}"
        for event in report_second["enforcement_events"]
    )
    assert any(
        event["scope"] == f"session:{first.session_id}" for event in guard.enforcement_events()
    )
    first.close()
    second.close()
