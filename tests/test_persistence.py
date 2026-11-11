"""SQLite persistence: restart recovery, crash recovery, ownership and failure handling."""

import sqlite3
from decimal import Decimal

import pytest

from owl_agent_budget_guard import (
    BudgetExceededError,
    Limits,
    Policies,
    SessionStoppedError,
)
from owl_agent_budget_guard.exceptions import PersistenceError, StoreOwnershipError
from owl_agent_budget_guard.store import SqliteStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "budget.db"


def _open(make_guard, db_path, **kwargs):
    return make_guard(store=SqliteStore(db_path), **kwargs)


def test_agent_and_session_state_survive_restart(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    session = guard.session("task", limits=Limits(max_cost=Decimal("0.80")))
    session_id = session.session_id
    with session.operation("0.20", llm_calls=1) as op:
        op.record_llm_call(reported_cost="0.12", input_tokens=10, output_tokens=5)
    guard.close()

    restarted = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    resumed = restarted.session("task", session_id=session_id)
    report = resumed.report()
    assert report["agent"]["budget"]["actual_spending"] == "0.12000000"
    assert report["session"]["budget"]["actual_spending"] == "0.12000000"
    assert report["session"]["usage"]["model_requests"] == 1
    assert report["session"]["usage"]["total_tokens"] == 15
    assert report["session"]["budget"]["limit"] == "0.80000000"
    resumed.close()
    restarted.close()


def test_stored_session_limit_still_enforced_after_restart(make_guard, db_path) -> None:
    guard = _open(
        make_guard,
        db_path,
        limits=Limits(max_cost=Decimal("10")),
        policies=Policies(on_budget_exceeded="raise"),
    )
    session = guard.session("capped", limits=Limits(max_cost=Decimal("0.50")))
    session_id = session.session_id
    session.reserve("0.10").reconcile("0.10")
    guard.close()

    restarted = _open(
        make_guard,
        db_path,
        limits=Limits(max_cost=Decimal("10")),
        policies=Policies(on_budget_exceeded="raise"),
    )
    resumed = restarted.session("capped", session_id=session_id)
    with pytest.raises(BudgetExceededError):
        resumed.reserve("0.50")
    resumed.reserve("0.40").release()
    restarted.close()


def test_stopped_session_stays_stopped_after_restart(make_guard, db_path) -> None:
    guard = _open(
        make_guard,
        db_path,
        limits=Limits(max_cost=Decimal("0.10")),
        policies=Policies(on_budget_exceeded="stop"),
    )
    session = guard.session("stopper")
    session_id = session.session_id
    with pytest.raises(BudgetExceededError):
        session.reserve("1")
    guard.close()

    restarted = _open(
        make_guard,
        db_path,
        limits=Limits(max_cost=Decimal("0.10")),
        policies=Policies(on_budget_exceeded="stop"),
    )
    resumed = restarted.session("stopper", session_id=session_id)
    assert resumed.state == "stopped"
    with pytest.raises(SessionStoppedError):
        resumed.reserve("0.01")
    restarted.close()


def test_enforcement_events_survive_restart(make_guard, db_path) -> None:
    guard = _open(
        make_guard,
        db_path,
        limits=Limits(max_cost=Decimal("0.10")),
        policies=Policies(on_budget_exceeded="raise"),
    )
    session = guard.session("events")
    with pytest.raises(BudgetExceededError):
        session.reserve("1")
    kinds_before = {event["kind"] for event in guard.enforcement_events()}
    guard.close()

    restarted = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("0.10")))
    kinds_after = {event["kind"] for event in restarted.enforcement_events()}
    assert "budget_rejected" in kinds_before
    assert kinds_before <= kinds_after
    restarted.close()


def test_open_reservation_is_recovered_as_charged_failure(make_guard, db_path) -> None:
    # Process 1 reserves and then "crashes" without settling.
    crashed = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    session = crashed.session("crashing")
    session_id = session.session_id
    session.reserve("0.30", llm_calls=1)

    # Process 2 starts, takes ownership, and recovers the open reservation.
    recovered = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    report = recovered.session("crashing", session_id=session_id).report()
    budget = report["agent"]["budget"]
    assert budget["reserved_amount"] == "0.00000000"
    assert budget["actual_spending"] == "0.30000000"
    assert budget["cost_by_source"]["estimated_fallback"] == "0.30000000"
    assert report["agent"]["usage"]["failed_operations"] == 1
    assert report["agent"]["usage"]["model_requests"] == 1
    kinds = [event["kind"] for event in recovered.enforcement_events()]
    assert "orphaned_reservation_recovered" in kinds
    recovered.close()
    crashed.close()  # the crashed process is gone; release its connection


def test_recovery_keeps_recorded_costs_and_uses_larger_value(make_guard, db_path) -> None:
    crashed = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("2")))
    session = crashed.session("partial")
    reservation = session.reserve("0.10", llm_calls=2)
    reservation.record_llm_call(reported_cost="0.25")  # more than the estimate
    session_id = session.session_id

    recovered = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("2")))
    budget = recovered.session("partial", session_id=session_id).report()["agent"]["budget"]
    # Recorded 0.25 is above the 0.10 estimate, so 0.25 is charged, not 0.10.
    assert budget["actual_spending"] == "0.25000000"
    assert budget["cost_by_source"]["reported"] == "0.25000000"
    assert budget["cost_by_source"]["estimated_fallback"] == "0.00000000"
    recovered.close()
    crashed.close()


def test_crashed_process_cannot_write_after_takeover(make_guard, db_path) -> None:
    old = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    old_session = old.session("old")
    old_session.reserve("0.10").release()

    new = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))  # takes ownership

    with pytest.raises(StoreOwnershipError):
        old.session("again")
    old.close()
    new.close()


def test_failed_write_refuses_further_work_and_persists_nothing(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    session = guard.session("failing")
    session_id = session.session_id

    real_save = guard._store.save

    def broken_save(**kwargs):
        raise PersistenceError("disk unavailable")

    guard._store.save = broken_save
    with pytest.raises(PersistenceError, match="disk unavailable"):
        session.reserve("0.50")
    guard._store.save = real_save

    with pytest.raises(PersistenceError, match="reopen the guard"):
        session.reserve("0.10")

    guard.close()
    restarted = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    report = restarted.session("failing", session_id=session_id).report()
    assert report["agent"]["budget"]["reserved_amount"] == "0.00000000"
    assert report["agent"]["budget"]["actual_spending"] == "0.00000000"
    restarted.close()


def test_save_is_atomic_and_rolls_back_on_error(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path, limits=Limits(max_cost=Decimal("1")))
    store = guard._store
    session = guard.session("atomic")
    session_id = session.session_id
    before = store.load()

    with pytest.raises(KeyError):
        store.save(
            owner=guard._instance_id,
            scopes={"agent:test-agent": {"actual_cost": "9"}},  # incomplete payload
            sessions=[],
            open_reservations={},
            events=[{"timestamp": "t", "scope": "s", "kind": "k"}],  # missing 'detail'
        )
    after = store.load()
    assert after.scopes == before.scopes
    assert after.events == before.events
    assert session_id in after.sessions
    guard.close()


def test_store_cannot_be_used_by_a_different_agent(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path, agent_name="first-agent")
    guard.close()
    with pytest.raises(PersistenceError, match="belongs to agent 'first-agent'"):
        _open(make_guard, db_path, agent_name="second-agent")


def test_schema_version_is_checked(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path)
    guard.close()
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(PersistenceError, match="schema version 99"):
        _open(make_guard, db_path)


def test_read_only_store_requires_existing_file(tmp_path) -> None:
    with pytest.raises(PersistenceError, match="no budget store"):
        SqliteStore(tmp_path / "missing.db", read_only=True)


def test_read_only_store_does_not_claim_ownership(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path)
    guard.session("live").reserve("0.01").release()
    inspector = SqliteStore(db_path, read_only=True)
    snapshot = inspector.load()
    assert snapshot.agent_name == "test-agent"
    inspector.close()
    # The live guard still owns the store and keeps writing.
    guard.session("still-live").reserve("0.01").release()
    guard.close()


def test_closed_session_stays_closed_after_restart(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path)
    session = guard.session("done")
    session_id = session.session_id
    session.close()
    guard.close()

    restarted = _open(make_guard, db_path)
    resumed = restarted.session("done", session_id=session_id)
    assert resumed.state == "closed"
    restarted.close()


def test_session_cannot_be_resumed_twice_in_one_guard(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path)
    session = guard.session("once")
    session_id = session.session_id
    with pytest.raises(PersistenceError, match="already attached"):
        guard.session("once", session_id=session_id)
    guard.close()


def test_resume_rejects_unknown_session_and_wrong_name(make_guard, db_path) -> None:
    guard = _open(make_guard, db_path)
    session = guard.session("named")
    session_id = session.session_id
    guard.close()

    restarted = _open(make_guard, db_path)
    with pytest.raises(PersistenceError, match="not in the budget store"):
        restarted.session("named", session_id="ses_nope")
    with pytest.raises(PersistenceError, match="named 'named'"):
        restarted.session("other", session_id=session_id)
    restarted.close()


def test_resume_requires_a_store(make_guard) -> None:
    guard = make_guard()
    with pytest.raises(PersistenceError, match="requires a guard with a store"):
        guard.session("x", session_id="ses_abc")


def test_guard_rejects_non_store_argument(make_guard) -> None:
    with pytest.raises(TypeError, match="SqliteStore"):
        make_guard(store="owl-budget.db")


def test_elapsed_time_is_restored_from_store(make_guard, db_path, clock) -> None:
    guard = _open(make_guard, db_path)
    session = guard.session("timed", limits=Limits(max_execution_time=100))
    session_id = session.session_id
    clock.advance(60)
    session.check()  # persists elapsed time, which is 60 seconds at this point
    guard.close()

    # A new process starts with a fresh clock. Only 40 seconds remain.
    clock.now = 5000.0
    restarted = _open(make_guard, db_path)
    resumed = restarted.session("timed", session_id=session_id)
    clock.advance(41)
    with pytest.raises(Exception, match="max_execution_time"):
        resumed.check()
    restarted.close()
