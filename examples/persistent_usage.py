"""Persistence example: budget survives a restart.

Runs two "process lifetimes" against one SQLite file in a temporary directory.
The first opens the guard, spends some budget, and exits. The second opens the
same file, resumes the session by id, and sees the earlier spending.

Run from the repository root:

    python examples/persistent_usage.py
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

from owl_agent_budget_guard import BudgetGuard, Limits, ModelPricing
from owl_agent_budget_guard.store import SqliteStore

PRICING = ModelPricing(
    model="demo-model",
    input_per_million=Decimal("1.00"),
    output_per_million=Decimal("5.00"),
    currency="USD",
    source="illustrative prices for this example only",
)


def open_guard(db_path: Path) -> BudgetGuard:
    return BudgetGuard(
        agent_name="persistent-example",
        currency="USD",
        limits=Limits(max_cost=Decimal("1.00")),
        pricing=[PRICING],
        store=SqliteStore(db_path),
    )


def first_run(db_path: Path) -> str:
    with open_guard(db_path) as guard:
        session = guard.session("nightly-analysis")
        with session.operation("0.05", llm_calls=1) as op:
            op.record_llm_call(model="demo-model", input_tokens=4000, output_tokens=1000)
        print(f"first run : spent {session.report()['session']['budget']['actual_spending']} USD")
        return session.session_id


def second_run(db_path: Path, session_id: str) -> None:
    with open_guard(db_path) as guard:
        session = guard.session("nightly-analysis", session_id=session_id)
        report = session.report()
        print(
            f"second run: resumed with {report['session']['budget']['actual_spending']} USD spent"
        )
        with session.operation("0.05", llm_calls=1) as op:
            op.record_llm_call(model="demo-model", input_tokens=4000, output_tokens=1000)
        report = session.report()
        print(f"second run: now {report['session']['budget']['actual_spending']} USD spent")
        print(
            f"            remaining agent budget: "
            f"{report['agent']['budget']['remaining_available']} USD"
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "budget.db"
        session_id = first_run(db_path)
        second_run(db_path, session_id)


if __name__ == "__main__":
    main()
