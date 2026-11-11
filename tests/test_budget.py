"""Budget engine: reservations, settlement, admission and policies."""

from decimal import Decimal

import pytest

from owl_agent_budget_guard import (
    ApprovalRequiredError,
    BudgetExceededError,
    InvalidValueError,
    Limits,
    Policies,
    ReservationError,
    ReservationOverrunError,
    SessionClosedError,
    SessionStoppedError,
)


def _remaining(session_report: dict, scope: str = "session") -> str | None:
    return session_report[scope]["budget"]["remaining_available"]


def test_initial_budget_equals_limit(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("5")))
    with guard.session(unique_name) as session:
        report = session.report()
    assert report["agent"]["budget"]["limit"] == "5.00000000"
    assert report["agent"]["budget"]["remaining_available"] == "5.00000000"
    assert report["session"]["budget"]["limit"] is None
    assert report["session"]["budget"]["remaining_available"] is None


def test_reservation_reduces_available_budget_before_settlement(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.30")
        report = session.report()
        assert report["agent"]["budget"]["reserved_amount"] == "0.30000000"
        assert report["agent"]["budget"]["actual_spending"] == "0.00000000"
        assert report["agent"]["budget"]["remaining_available"] == "0.70000000"
        assert reservation.is_open
        reservation.release()


def test_reconcile_charges_actual_and_releases_unused_reservation(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.40")
        reservation.reconcile("0.15")
        report = session.report()
    assert reservation.state == "settled"
    assert report["agent"]["budget"]["actual_spending"] == "0.15000000"
    assert report["agent"]["budget"]["reserved_amount"] == "0.00000000"
    assert report["agent"]["budget"]["remaining_available"] == "0.85000000"
    assert report["agent"]["budget"]["cost_by_source"]["reconciled"] == "0.15000000"


def test_release_refunds_full_reservation(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.60")
        reservation.release()
        report = session.report()
    assert reservation.state == "released"
    assert report["agent"]["budget"]["actual_spending"] == "0.00000000"
    assert report["agent"]["budget"]["remaining_available"] == "1.00000000"
    assert report["session"]["usage"]["failed_operations"] == 1


def test_double_reconcile_is_rejected(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.20")
        reservation.reconcile("0.10")
        with pytest.raises(ReservationError, match="already settled"):
            reservation.reconcile("0.10")
        with pytest.raises(
            ReservationError, match="cannot be recorded, reconciled or released again"
        ):
            reservation.release()
        assert session.report()["agent"]["budget"]["actual_spending"] == "0.10000000"


def test_release_after_reconcile_is_rejected(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.20")
        reservation.reconcile("0.05")
        with pytest.raises(ReservationError):
            reservation.release()


def test_recording_after_settlement_is_rejected(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.20")
        reservation.release()
        with pytest.raises(ReservationError):
            reservation.record_tool_call(cost=0)


def test_reconcile_above_reservation_records_cost_then_raises(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.10")
        with pytest.raises(ReservationOverrunError):
            reservation.reconcile("0.25")
        report = session.report()
    assert report["agent"]["budget"]["actual_spending"] == "0.25000000"
    assert report["agent"]["budget"]["remaining_available"] == "0.75000000"
    assert any(e["kind"] == "reservation_overrun" for e in report["enforcement_events"])


def test_reconcile_below_recorded_cost_is_rejected(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        reservation = session.reserve("0.50", tool_calls=1)
        reservation.record_tool_call(cost="0.20")
        with pytest.raises(InvalidValueError, match="lower than"):
            reservation.reconcile("0.10")
        assert reservation.is_open
        reservation.reconcile("0.30")
    assert session.state == "closed"


def test_budget_exhaustion_rejects_reservation(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("0.50")))
    with guard.session(unique_name) as session:
        session.reserve("0.50").reconcile("0.50")
        with pytest.raises(BudgetExceededError, match="exhausted"):
            session.reserve("0.01")


def test_reservation_that_exceeds_available_is_rejected_without_side_effects(
    make_guard, unique_name
) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")), policies=Policies(on_budget_exceeded="raise")
    )
    with guard.session(unique_name) as session:
        session.reserve("0.80")
        with pytest.raises(BudgetExceededError) as info:
            session.reserve("0.30")
        assert info.value.requested == Decimal("0.30000000")
        assert info.value.available == Decimal("0.20000000")
        report = session.report()
    assert report["agent"]["budget"]["reserved_amount"] == "0.80000000"
    assert report["session"]["state"] == "active"


def test_zero_remaining_rejects_even_zero_cost_operations(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.10")), policies=Policies(on_budget_exceeded="raise")
    )
    with guard.session(unique_name) as session:
        session.reserve("0.10").reconcile("0.10")
        with pytest.raises(BudgetExceededError, match="exhausted"):
            session.reserve("0")
        with pytest.raises(BudgetExceededError, match="exhausted"):
            session.check()


@pytest.mark.parametrize("bad_cost", [-1, "-0.1", "NaN", True, None, "1.000000001"])
def test_invalid_monetary_inputs_are_rejected(make_guard, unique_name, bad_cost) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(InvalidValueError):
            session.reserve(bad_cost)


def test_estimated_cost_is_required(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:
        with pytest.raises(TypeError):
            session.reserve()  # type: ignore[call-arg]


def test_stop_policy_stops_session_after_rejection(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.10")), policies=Policies(on_budget_exceeded="stop")
    )
    with guard.session(unique_name) as session:
        with pytest.raises(BudgetExceededError, match="Session stopped"):
            session.reserve("0.50")
        assert session.state == "stopped"
        with pytest.raises(SessionStoppedError):
            session.reserve("0.01")
        with pytest.raises(SessionStoppedError):
            session.check()


def test_raise_policy_keeps_session_usable(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.10")), policies=Policies(on_budget_exceeded="raise")
    )
    with guard.session(unique_name) as session:
        with pytest.raises(BudgetExceededError):
            session.reserve("0.50")
        assert session.state == "active"
        session.reserve("0.05").release()


def test_pause_policy_raises_approval_required_and_pauses(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.10")), policies=Policies(on_budget_exceeded="pause")
    )
    with guard.session(unique_name) as session:
        with pytest.raises(ApprovalRequiredError) as info:
            session.reserve("0.50")
        assert info.value.state == "approval_required"
        assert session.state == "paused"
        with pytest.raises(ApprovalRequiredError):
            session.reserve("0.01")


def test_operation_cap_rejects_large_estimate_without_stopping(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("5"), max_operation_cost=Decimal("0.50")),
        policies=Policies(on_budget_exceeded="stop"),
    )
    with guard.session(unique_name) as session:
        with pytest.raises(BudgetExceededError, match="max_operation_cost"):
            session.reserve("0.75")
        assert session.state == "active"
        session.reserve("0.50").release()


def test_agent_limit_is_shared_across_sessions(make_guard) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")), policies=Policies(on_budget_exceeded="raise")
    )
    first = guard.session("first")
    second = guard.session("second")
    first.reserve("0.70").reconcile("0.70")
    with pytest.raises(BudgetExceededError) as info:
        second.reserve("0.50")
    assert info.value.scope.startswith("agent:")
    first.close()
    second.close()


def test_session_limit_is_stricter_than_agent_limit(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("10")), policies=Policies(on_budget_exceeded="raise")
    )
    with guard.session(unique_name, limits=Limits(max_cost=Decimal("0.20"))) as session:
        session.reserve("0.15").reconcile("0.15")
        with pytest.raises(BudgetExceededError) as info:
            session.reserve("0.10")
        assert info.value.scope.startswith("session:")


def test_sessions_do_not_share_session_level_accounting(make_guard) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("10")))
    first = guard.session("first", limits=Limits(max_cost=Decimal("1")))
    second = guard.session("second", limits=Limits(max_cost=Decimal("1")))
    first.reserve("0.90").reconcile("0.90")
    second_report = second.report()
    assert second_report["session"]["budget"]["actual_spending"] == "0.00000000"
    assert second_report["session"]["budget"]["remaining_available"] == "1.00000000"
    assert second_report["agent"]["budget"]["actual_spending"] == "0.90000000"
    first.close()
    second.close()


def test_closing_session_releases_open_reservations(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    session = guard.session(unique_name)
    reservation = session.reserve("0.80")
    session.close()
    assert reservation.state == "released"
    assert guard.session("after").report()["agent"]["budget"]["reserved_amount"] == "0.00000000"


def test_closed_session_rejects_work(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    session = guard.session(unique_name)
    session.close()
    session.close()  # idempotent
    with pytest.raises(SessionClosedError):
        session.reserve("0.10")
    with pytest.raises(SessionClosedError):
        session.check()


def test_enforcement_events_are_recorded(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.10")), policies=Policies(on_budget_exceeded="raise")
    )
    with guard.session(unique_name) as session:
        with pytest.raises(BudgetExceededError):
            session.reserve("1")
    kinds = {event["kind"] for event in guard.enforcement_events()}
    assert "budget_rejected" in kinds
