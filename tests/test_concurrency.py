"""Concurrency: reservations must never overspend, even when many threads race."""

import threading
from decimal import Decimal

from owl_agent_budget_guard import BudgetExceededError, Limits, OwlBudgetError, Policies

THREADS = 32


def _run_threads(target, count: int) -> list[BaseException | None]:
    barrier = threading.Barrier(count)
    results: list[BaseException | None] = [None] * count

    def wrapper(index: int) -> None:
        barrier.wait()
        try:
            target(index)
        except BaseException as exc:  # recorded for assertions, not swallowed silently
            results[index] = exc

    threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_concurrent_reservations_never_exceed_budget(make_guard) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.50")), policies=Policies(on_budget_exceeded="raise")
    )
    session = guard.session("race")
    accepted: list[int] = []
    accepted_lock = threading.Lock()

    def attempt(index: int) -> None:
        try:
            session.reserve("0.10")
        except BudgetExceededError:
            return
        with accepted_lock:
            accepted.append(index)

    results = _run_threads(attempt, THREADS)
    assert all(result is None for result in results)
    assert len(accepted) == 5  # exactly 0.50 / 0.10
    report = session.report()
    assert report["agent"]["budget"]["reserved_amount"] == "0.50000000"
    assert report["agent"]["budget"]["remaining_available"] == "0.00000000"
    session.close()


def test_concurrent_operations_settle_to_exact_total(make_guard) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("100")))
    session = guard.session("many-ops")

    def run_operation(_index: int) -> None:
        with session.operation("0.10", llm_calls=1) as op:
            op.record_llm_call(reported_cost="0.03", input_tokens=10, output_tokens=10)

    results = _run_threads(run_operation, THREADS)
    assert all(result is None for result in results)
    report = session.report()
    assert report["agent"]["budget"]["actual_spending"] == "0.96000000"
    assert report["agent"]["budget"]["reserved_amount"] == "0.00000000"
    assert report["session"]["usage"]["successful_operations"] == THREADS
    assert report["session"]["usage"]["model_requests"] == THREADS
    assert report["session"]["usage"]["total_tokens"] == 20 * THREADS
    session.close()


def test_concurrent_sessions_share_agent_budget_safely(make_guard) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("1")), policies=Policies(on_budget_exceeded="raise")
    )
    sessions = [guard.session(f"s{i}") for i in range(THREADS)]
    granted: list[int] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        try:
            sessions[index].reserve("0.07").reconcile("0.07")
        except BudgetExceededError:
            return
        with lock:
            granted.append(index)

    _run_threads(attempt, THREADS)
    # 1 / 0.07 = 14.28 -> 14 admissions fit.
    assert len(granted) == 14
    assert guard.session("check").report()["agent"]["budget"]["actual_spending"] == "0.98000000"
    for session in sessions:
        session.close()


def test_stop_policy_under_contention_stops_session_once(make_guard) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.05")), policies=Policies(on_budget_exceeded="stop")
    )
    session = guard.session("stop-race")

    rejected: list[type[BaseException]] = []
    lock = threading.Lock()

    def attempt(_index: int) -> None:
        try:
            session.reserve("0.10")
        except (
            OwlBudgetError
        ) as exc:  # first is BudgetExceededError; later ones SessionStoppedError
            with lock:
                rejected.append(type(exc))

    results = _run_threads(attempt, THREADS)
    assert all(result is None for result in results)
    assert len(rejected) == THREADS
    assert rejected.count(BudgetExceededError) == 1
    assert session.state == "stopped"
    stops = [e for e in guard.enforcement_events() if e["kind"] == "session_stopped"]
    assert len(stops) == 1
    session.close()
