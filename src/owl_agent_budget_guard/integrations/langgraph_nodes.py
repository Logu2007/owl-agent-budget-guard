"""LangGraph integration: budget each node run as one operation.

Wrap a node function with :func:`budgeted_node`. Every time the node runs, the
wrapper reserves ``estimated_cost`` before the node body executes, then settles
the reservation when the node returns or raises. A node that needs to record
usage declares a keyword-only ``budget`` parameter and calls ``budget.record_llm_call``
or ``budget.record_tool_call``.

If the budget does not fit, the node body never runs. The error
(:class:`BudgetExceededError`, :class:`LimitExceededError`, and so on) propagates
out of ``graph.invoke`` or ``graph.stream``, and the run stops. LangGraph does not
catch it.

Example::

    from langgraph.graph import END, START, StateGraph
    from owl_agent_budget_guard.integrations.langgraph_nodes import budgeted_node

    @budgeted_node(session, estimated_cost="0.02", llm_calls=1)
    def analyse(state, *, budget):
        usage = my_model(state["question"])
        budget.record_llm_call(model="my-model",
                               input_tokens=usage.input, output_tokens=usage.output)
        return {"answer": usage.text}

    graph = StateGraph(State)
    graph.add_node("analyse", analyse)
    graph.add_edge(START, "analyse")
    graph.add_edge("analyse", END)
    app = graph.compile()

Each node run needs its own ``estimated_cost``. Budget for a whole graph run by
using one session and ``session.check()`` beforehand, or by giving the session
limits.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from owl_agent_budget_guard.session import Session

F = TypeVar("F", bound=Callable[..., Any])


def _accepts_parameter(fn: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def _call_kwargs(
    budget: Any,
    config: Any,
    wants_budget: bool,
    wants_config: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if wants_budget:
        kwargs["budget"] = budget
    if wants_config:
        kwargs["config"] = config
    return kwargs


def budgeted_node(
    session: Session,
    estimated_cost: object,
    *,
    llm_calls: int = 0,
    tool_calls: int = 0,
) -> Callable[[F], F]:
    """Decorator that runs a LangGraph node as one budget operation.

    Args:
        session: Session that pays for the node runs.
        estimated_cost: Estimate reserved before each run. Required.
        llm_calls: Maximum model requests a single run may make.
        tool_calls: Maximum tool executions a single run may make.

    The wrapped node may be a sync or async function. It may take a keyword-only
    ``budget`` parameter (the :class:`Reservation`) and a ``config`` parameter
    (the LangGraph run config).
    """

    def decorator(node: F) -> F:
        wants_budget = _accepts_parameter(node, "budget")
        wants_config = _accepts_parameter(node, "config")

        if inspect.iscoroutinefunction(node):

            @functools.wraps(node)
            async def async_wrapper(state: Any, config: Any = None) -> Any:
                with session.operation(
                    estimated_cost, llm_calls=llm_calls, tool_calls=tool_calls
                ) as budget:
                    kwargs = _call_kwargs(budget, config, wants_budget, wants_config)
                    return await node(state, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(node)
        def sync_wrapper(state: Any, config: Any = None) -> Any:
            with session.operation(
                estimated_cost, llm_calls=llm_calls, tool_calls=tool_calls
            ) as budget:
                kwargs = _call_kwargs(budget, config, wants_budget, wants_config)
                return node(state, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
