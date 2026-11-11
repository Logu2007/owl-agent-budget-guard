"""Core SDK example: budget an agent run that makes model and tool calls.

The "model" below is a local simulation. It returns made-up token counts so the
example runs without any API key. Replace ``simulated_model_call`` with your
provider client, and pass the usage numbers that the provider returns.

Run from the repository root after installing the package:

    python examples/basic_usage.py
"""

from __future__ import annotations

from decimal import Decimal

from owl_agent_budget_guard import (
    BudgetGuard,
    Limits,
    ModelPricing,
    OwlBudgetError,
    Policies,
)

# Illustrative prices only. Use your provider's current, verified prices.
DEMO_MODEL = ModelPricing(
    model="demo-model",
    input_per_million=Decimal("1.00"),
    output_per_million=Decimal("5.00"),
    currency="USD",
    source="illustrative prices for this example only",
)


def simulated_model_call(prompt: str) -> dict[str, int]:
    """Stand-in for a provider call. Returns the usage a provider would report."""
    return {"input_tokens": len(prompt) * 10, "output_tokens": 200}


def simulated_tool_call(name: str) -> str:
    return f"result of {name}"


def analyse(guard: BudgetGuard, topic: str) -> None:
    with guard.session(f"analyse {topic}") as session:
        # Declare exactly the calls this operation makes: one model call, one tool call.
        with session.operation("0.02", llm_calls=1, tool_calls=1) as op:
            usage = simulated_model_call(f"Summarise {topic}")
            op.record_llm_call(
                model="demo-model",
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
            simulated_tool_call("search")
            op.record_tool_call(cost=0)  # the tool is free; zero must be explicit

        spent = session.report()["session"]["budget"]["actual_spending"]
        print(f"  {topic}: session actual spending {spent} USD")


def main() -> None:
    guard = BudgetGuard(
        agent_name="example-agent",
        currency="USD",
        limits=Limits(max_cost=Decimal("0.05"), max_llm_calls=10, max_tool_calls=10),
        policies=Policies(on_budget_exceeded="raise", unknown_cost="block"),
        pricing=[DEMO_MODEL],
    )

    print("Run 1 (within budget):")
    analyse(guard, "first topic")

    print("\nRun 2 (keeps going until a limit stops it):")
    try:
        for index in range(20):
            analyse(guard, f"topic {index}")
    except OwlBudgetError as exc:
        # BudgetExceededError (money) or LimitExceededError (requests/tokens/time).
        print(f"Stopped by the guard: {type(exc).__name__}: {exc}")

    print("\nEnforcement events:")
    for event in guard.enforcement_events():
        print(f"  {event['kind']}: {event['detail']}")


if __name__ == "__main__":
    main()
