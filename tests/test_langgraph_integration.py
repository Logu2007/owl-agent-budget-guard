"""LangGraph integration, run against a real compiled graph."""

import asyncio
from decimal import Decimal
from typing import TypedDict

import pytest

from owl_agent_budget_guard import BudgetExceededError, Limits, Policies
from owl_agent_budget_guard.integrations.langgraph_nodes import budgeted_node

langgraph_graph = pytest.importorskip("langgraph.graph")
from langgraph.graph import END, START, StateGraph  # noqa: E402


class State(TypedDict):
    question: str
    answer: str


def _single_node_app(node):
    graph = StateGraph(State)
    graph.add_node("analyse", node)
    graph.add_edge(START, "analyse")
    graph.add_edge("analyse", END)
    return graph.compile()


def test_node_runs_as_budget_operation_and_records_usage(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:

        @budgeted_node(session, "0.10", llm_calls=1)
        def analyse(state, *, budget):
            budget.record_llm_call(model="test-model", input_tokens=1000, output_tokens=500)
            return {"answer": f"answer to {state['question']}"}

        result = _single_node_app(analyse).invoke({"question": "why", "answer": ""})
        report = session.report()
    assert result["answer"] == "answer to why"
    assert report["session"]["budget"]["actual_spending"] == "0.00350000"
    assert report["session"]["usage"]["successful_operations"] == 1


def test_config_is_passed_when_the_node_accepts_it(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    seen = {}
    with guard.session(unique_name) as session:

        @budgeted_node(session, "0", tool_calls=1)
        def analyse(state, *, budget, config):
            seen["config_type"] = type(config).__name__
            budget.record_tool_call(cost=0)
            return {"answer": "ok"}

        _single_node_app(analyse).invoke({"question": "q", "answer": ""})
    assert seen["config_type"] == "dict"


def test_node_without_budget_parameter_still_settles(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:

        @budgeted_node(session, "0.25")
        def analyse(state):
            return {"answer": "no usage recorded"}

        # Nothing was recorded, so the block policy rejects the run after settlement.
        with pytest.raises(Exception, match="without recording any cost"):
            _single_node_app(analyse).invoke({"question": "q", "answer": ""})
        assert session.report()["session"]["budget"]["actual_spending"] == "0.25000000"


def test_budget_exhaustion_stops_the_graph_before_the_node_runs(make_guard, unique_name) -> None:
    guard = make_guard(
        limits=Limits(max_cost=Decimal("0.05")), policies=Policies(on_budget_exceeded="raise")
    )
    ran = []
    with guard.session(unique_name) as session:

        @budgeted_node(session, "0.10")
        def analyse(state, *, budget):
            ran.append(True)
            budget.record_tool_call(cost=0)
            return {"answer": "should not happen"}

        with pytest.raises(BudgetExceededError):
            _single_node_app(analyse).invoke({"question": "q", "answer": ""})
    assert ran == []


def test_async_node_is_supported(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:

        @budgeted_node(session, "0.10", llm_calls=1)
        async def analyse(state, *, budget):
            budget.record_llm_call(model="test-model", input_tokens=0, output_tokens=1000)
            return {"answer": "async ok"}

        result = asyncio.run(_single_node_app(analyse).ainvoke({"question": "q", "answer": ""}))
        assert result["answer"] == "async ok"
        assert session.report()["session"]["budget"]["actual_spending"] == "0.00500000"


def test_two_node_graph_charges_each_run_separately(make_guard, unique_name) -> None:
    guard = make_guard(limits=Limits(max_cost=Decimal("1")))
    with guard.session(unique_name) as session:

        @budgeted_node(session, "0.10", tool_calls=1)
        def first(state, *, budget):
            budget.record_tool_call(cost="0.02")
            return {"answer": "step one"}

        @budgeted_node(session, "0.10", tool_calls=1)
        def second(state, *, budget):
            budget.record_tool_call(cost="0.03")
            return {"answer": state["answer"] + " and two"}

        graph = StateGraph(State)
        graph.add_node("first", first)
        graph.add_node("second", second)
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        result = graph.compile().invoke({"question": "q", "answer": ""})
        report = session.report()
    assert result["answer"] == "step one and two"
    assert report["session"]["budget"]["actual_spending"] == "0.05000000"
    assert report["session"]["usage"]["successful_operations"] == 2
