# OWL Agent Budget Guard

OWL Agent Budget Guard is a local Python library and CLI that enforces spending,
token, request, tool and time limits on AI agent runs. It runs in your process,
needs no account, server or database, and works on whatever usage numbers your
code reports to it.

Status: **0.1.0, alpha.** The implementation is tested locally, but it has not
been used in production and its API may change.

Budget state can be persisted to a SQLite file so it survives restarts. Optional
integrations cover the OpenAI SDK and LangGraph. See [Persistence](#persistence)
and [Framework integrations](#framework-integrations).

---

## Problems it addresses

- An agent loops and keeps calling a model or a tool.
- A single task unexpectedly spends much more than expected.
- Several operations run concurrently and each one checks the budget before
  spending, so together they overspend.
- A run needs a readable record of what it spent, and why each amount was charged.

## Features

- **Budget engine.** Agent-wide and per-session cost limits, operation caps, and
  the formula `available = limit - actual - reserved`. Admission is atomic, so
  concurrent reservations cannot overspend.
- **Reservations.** Reserve an estimate before work starts, then reconcile to the
  actual cost, release it, or let the context manager settle it.
- **Usage tracking.** Input, output and cached tokens; model requests; tool
  executions; successful and failed operations; cost by source.
- **Limits.** Total tokens per scope, per-call input and output tokens, model
  requests, tool executions, and monotonic execution time.
- **Policies.** `stop`, `raise` or `pause` when the budget is exhausted;
  `block` or `estimate` when a cost is unknown.
- **Pricing registry.** Prices are configured by you, never bundled. Arithmetic
  uses `Decimal`, and costs round up to eight places.
- **YAML configuration.** Safe loading, duplicate-key rejection, unknown-field
  rejection, and clear error messages.
- **Persistence.** Optional SQLite store: budget state, sessions, open
  reservations and events survive restarts. Open reservations left by a crash are
  charged conservatively on the next start. See [Persistence](#persistence).
- **Integrations.** Optional OpenAI SDK helpers that record real response usage,
  and a LangGraph node wrapper that budgets each node run.
- **CLI.** `owl-budget init | validate | status | report | pricing | version`.
- **Reports.** JSON-compatible dictionaries with consistent field names, and a
  readable text view. Estimates and actual amounts are reported separately.

## Architecture

```
BudgetGuard ─── agent-wide scope (limits + counters), enforcement event log, PricingRegistry
   └── Session ─── session scope (its own limits + counters)
          └── Reservation ─── estimate held at admission; usage recorded; settled once
```

| Module | Responsibility |
| --- | --- |
| `guard.py` | `BudgetGuard`: owns the agent scope, the lock and the event log |
| `session.py` | `Session`: admission checks, policies, operation context manager, report |
| `reservation.py` | `Reservation`: usage recording and single settlement |
| `ledger.py` | Scope counters and pure check helpers (called under the lock) |
| `limits.py` | `Limits` and `Policies` dataclasses with validation |
| `pricing.py` | `ModelPricing` and `PricingRegistry`, cost calculation |
| `money.py` | Decimal conversion, eight-place precision, formatting |
| `config.py` | Safe YAML loading, Pydantic validation, conversion to SDK objects |
| `report.py` | Report dictionary and text rendering |
| `store.py` | `SqliteStore`: schema, transactions, ownership fencing, serialization |
| `store_report.py` | Read-only reports from a stored budget, for the CLI |
| `integrations/openai_sdk.py` | Record usage from OpenAI responses (no `openai` import) |
| `integrations/langgraph_nodes.py` | `budgeted_node` decorator for LangGraph nodes |
| `cli.py` | Typer commands |

Every admission and settlement runs under one re-entrant lock per guard, so
threads in one process share consistent accounting.

## Installation

Requires Python 3.11 or newer.

```bash
git clone <this repository>
cd owl-agent-budget-guard
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies: `pydantic`, `PyYAML`, `typer`. The core SDK imports no AI
provider SDK and no agent framework.

Optional extras:

```bash
pip install -e ".[openai]"      # the openai package, for making calls
pip install -e ".[langgraph]"   # langgraph, for the node wrapper
pip install -e ".[dev]"         # tests and lint; includes both of the above
```

Tested with `openai` 3.46.0 and `langgraph` 1.2.20.

## Quick start

```python
from decimal import Decimal
from owl_agent_budget_guard import BudgetGuard, Limits, ModelPricing, Policies

guard = BudgetGuard(
    agent_name="website-analyser",
    currency="USD",
    limits=Limits(max_cost=Decimal("5.00"), max_llm_calls=50, max_execution_time=600),
    policies=Policies(on_budget_exceeded="stop", unknown_cost="block"),
    pricing=[
        # Illustrative prices only. Use your provider's verified prices.
        ModelPricing(model="example-model", input_per_million=Decimal("1.00"),
                     output_per_million=Decimal("5.00"), currency="USD"),
    ],
)

with guard.session("website-analysis") as session:
    session.check()  # raises if the budget is already exhausted

    with session.operation("0.10", llm_calls=1) as op:  # reserve 0.10 before running
        response = my_model_client.generate(prompt)      # your code
        op.record_llm_call(
            model="example-model",
            input_tokens=response.usage.input_tokens,    # from the provider
            output_tokens=response.usage.output_tokens,
        )
    # Leaving the block charges the calculated cost and releases the unused estimate.

    print(session.report()["session"]["budget"])
```

Runnable version without any API key: `python examples/basic_usage.py`.

## Configuration

Create a starter file with `owl-budget init`, or copy
`examples/owl-budget.example.yaml`.

```yaml
version: 1                  # required, must be 1

agent:
  name: owl-security-agent  # required; letters, digits, spaces and . _ : / -

budget:
  max_cost: 5.00            # required; agent-wide cap, >= 0, at most 8 decimal places
  currency: USD             # default USD; three-letter code

store:                      # optional; omit for in-memory state
  path: owl-budget.db       # relative paths resolve against the working directory

limits:                     # all optional; omitted means not limited
  max_cost_per_operation: 0.50
  max_llm_calls: 50
  max_tool_calls: 100
  max_tokens: 100000
  max_input_tokens_per_call: 20000
  max_output_tokens_per_call: 10000
  max_execution_time: 600   # seconds; positive; monotonic clock

policies:
  on_budget_exceeded: stop  # stop (default) | raise | pause
  unknown_cost: block       # block (default) | estimate
  allow_budget_extension: false   # must be false; automatic increases are not supported

pricing:
  models:                   # empty by default; no prices are bundled
    example-model:
      currency: USD         # must match budget.currency
      input_per_million: 1.00
      output_per_million: 5.00
      cached_input_per_million: 0.10   # optional
      source: "https://example.com/pricing, verified 2026-01-01"
```

Validation rejects unknown keys at every level, negative or zero limits,
wrong types, booleans used as numbers, duplicate keys, pricing in a different
currency from the budget, `allow_budget_extension: true`, files over 256 KiB,
and any YAML construct that is not plain data. Error messages give the field
path and rule, and never echo the file's values.

Config-file limits apply to the agent-wide scope. Session-level limits are set
in code with `guard.session(name, limits=Limits(...))`.

When `store` is set, the CLI reads the stored budget, and `build_guard(config)`
opens the store for a running guard.

## Python SDK

### Objects

- `BudgetGuard(agent_name, currency, limits, policies, pricing)`: agent-wide scope.
- `guard.session(name, limits=None)`: a session (task). Use it as a context manager.
- `session.check()`: raises if the session cannot accept work. Reserves nothing.
- `session.reserve(estimated_cost, llm_calls=0, tool_calls=0)`: returns a
  `Reservation`. Use this form when the operation does not fit a block.
- `session.operation(estimated_cost, llm_calls=0, tool_calls=0)`: context manager
  that yields a `Reservation` and settles it on exit.
- `Reservation.record_llm_call(model=None, input_tokens=None, output_tokens=None, cached_input_tokens=0, reported_cost=None)`
- `Reservation.record_tool_call(cost=None)`: `cost` is required. Use `0` for a free tool.
- `Reservation.reconcile(actual_cost=None)`: explicit settlement.
- `Reservation.release()`: abandon the operation.
- `session.report()`: JSON-compatible dictionary (see *Reports*).
- `guard.enforcement_events()`: list of events recorded by the guard.
- `guard.register_pricing(ModelPricing(...))`: add prices in code.

`estimated_cost` is always required. An operation cannot be admitted without an
estimate, and an estimate is never assumed to be zero.

### Money

Amounts are `Decimal`. Strings (`"0.10"`), ints and floats are accepted. Floats
go through their shortest repr, so `0.1` becomes `Decimal("0.1")`. Amounts must
be non-negative, finite, and have at most eight decimal places. Anything else
raises `InvalidValueError`; values are never silently rounded on input.

### Usage recording

Pass the numbers your provider returns. The guard does not read them from the
provider.

- Cost source precedence for one model call: `reported_cost` if given, otherwise
  a cost calculated from registered pricing and the token counts, otherwise the
  cost is **unknown**.
- Cached tokens are a subset of `input_tokens`. They are billed at
  `cached_input_per_million` when that price is configured.
- A tool call without `cost=` is unknown. Pass `cost=0` for a free tool.

### Settlement rules

The guard never treats an unknown cost as zero.

| Situation | Charged | Other effects |
| --- | --- | --- |
| Block exits normally, `reconcile(x)` called | `x` (must be at least the recorded cost) | Residual above recorded costs is source `reconciled` |
| Block exits normally, costs recorded and all known | Sum of recorded costs | Unused estimate released |
| Block exits normally, cost unknown or nothing recorded | `max(known, estimate)`, source `estimated_fallback` | `UnknownCostError` under `unknown_cost=block`; event under `estimate` |
| Block raises, nothing recorded | Nothing | Reservation released, error propagates unchanged |
| Block raises, costs recorded | Recorded costs (or `max(known, estimate)` if any cost is unknown) | Original error propagates unchanged |
| Actual cost above the estimate | Actual cost | `ReservationOverrunError` on normal exit; the spend is recorded |

A reservation is settled once. Reconciling or releasing it again raises
`ReservationError`. Closing a session releases any reservations still open.

### Request and tool counts

`llm_calls` and `tool_calls` declared at admission are reserved against the
limits. Recording a call moves it from reserved to used. Recording more calls
than were declared raises `LimitExceededError`. On a normal exit, unused
declarations are released. On failure, unused declarations count as used,
because the call may already have happened.

### Policies

Triggered when an operation's cost does not fit the remaining budget.

- `stop` (default): the operation is rejected with `BudgetExceededError`, and the
  session is stopped. Every later operation raises `SessionStoppedError`.
- `raise`: the operation is rejected with `BudgetExceededError`. The session stays
  usable for operations that fit.
- `pause`: the operation is rejected with `ApprovalRequiredError`
  (`state == "approval_required"`), and the session is paused. The SDK has no
  approval API in 0.1.0, so the application decides whether to start a new
  session with explicit limits.

Zero remaining budget rejects every operation, including those with a zero estimate.
Other limits (requests, tokens, time) always raise `LimitExceededError` or
`ExecutionTimeExceededError` and do not change the session state.

## Persistence

Pass a `SqliteStore` to keep budget state across restarts:

```python
from decimal import Decimal
from owl_agent_budget_guard import BudgetGuard, Limits
from owl_agent_budget_guard.store import SqliteStore


def open_guard():
    return BudgetGuard(
        agent_name="website-analyser",       # must match the stored agent
        currency="USD",                      # must match the stored currency
        limits=Limits(max_cost=Decimal("5")),
        store=SqliteStore("owl-budget.db"),
    )


guard = open_guard()
session = guard.session("nightly-run")
session_id = session.session_id              # keep this to resume later
# ... work ...
guard.close()

# After a restart:
guard = open_guard()
session = guard.session("nightly-run", session_id=session_id)
print(session.report()["session"]["budget"])
```

**What is stored:** agent and session counters, session state (a stopped session
stays stopped), session limits, open reservations, and the enforcement event log.
Agent limits come from code each time, so a changed configuration applies on the
next start. Session limits are stored with the session and used on resume, unless
you pass new ones.

**Writes are transactional.** Each state change is saved in one SQLite write
transaction (`BEGIN IMMEDIATE`), and the file uses WAL mode. If a write fails,
nothing from that change is committed, and the guard refuses further work with
`PersistenceError`. Reopen the guard to continue. Its in-memory state may be ahead
of the store, so the store is the source of truth.

**One owner at a time.** Opening a store claims ownership with a random instance
id. Every write checks that id inside its transaction. If a second process opens
the same file, the first guard's next write fails with `StoreOwnershipError`, and
that guard stops. Budgets are never silently shared between live guards.

**Crash recovery.** A reservation still open when the previous process stopped is
settled on the next start as a failed operation. It is charged the larger of its
recorded costs and its estimate, and an `orphaned_reservation_recovered` event is
recorded. Spending from a crash is never treated as zero. Its unused request
declarations count as used, as they do for any failed operation.

**Elapsed time** is stored at each write, so a resumed session continues from the
time it had used. Time between the last write and a crash is not counted.

**Schema.** The store records a schema version. A file with an unknown version is
rejected rather than guessed at.

**Inspection.** `owl-budget status` and `owl-budget report` read the store
read-only (see [CLI](#cli)).

## Framework integrations

### OpenAI SDK

`owl_agent_budget_guard.integrations.openai_sdk` records the usage the SDK returns.
It reads fields by name, so the core package does not import `openai`:

```python
from owl_agent_budget_guard.integrations.openai_sdk import create_and_record

with session.operation("0.02", llm_calls=1) as op:
    response = create_and_record(
        op, client.chat.completions.create,
        model="gpt-4o-mini", messages=messages,   # all kwargs go to the SDK unchanged
    )
```

- `record_openai_usage(op, response)` records a response you already have.
- `acreate_and_record` is the async form, for `AsyncOpenAI`.
- Both Chat Completions (`prompt_tokens`, `completion_tokens`) and Responses
  (`input_tokens`, `output_tokens`) shapes are supported, including cached tokens.
- A response with no usage is recorded as unknown, so the `unknown_cost` policy
  applies.
- Cache-write tokens are billed at the normal input rate. Register prices that match
  what you pay, or pass `reported_cost`.
- Pricing uses the request's `model` name. If the registered name differs (for
  example a dated snapshot), pass `pricing_model="..."`.
- To cap output before a call, pass `max_completion_tokens` (Chat Completions) or
  `max_output_tokens` (Responses). `max_output_tokens_per_call` is checked only after
  the call.

### LangGraph

`owl_agent_budget_guard.integrations.langgraph_nodes.budgeted_node` makes each node
run one budget operation:

```python
from owl_agent_budget_guard.integrations.langgraph_nodes import budgeted_node


@budgeted_node(session, "0.02", llm_calls=1, tool_calls=1)
def analyse(state, *, budget):          # `budget` is the Reservation
    result = my_model(state["question"])
    budget.record_llm_call(model="my-model",
                           input_tokens=result.input, output_tokens=result.output)
    return {"answer": result.text}


graph = StateGraph(State)
graph.add_node("analyse", analyse)
```

- A reservation is made before the node body runs. If the budget does not fit, the
  body never runs, and the error propagates out of `invoke`, `stream`, or `ainvoke`.
  LangGraph does not catch it.
- The node settles when it returns or raises. Sync and `async` nodes are both
  supported. A `config` parameter is passed through when the node declares one.
- Each node run has its own estimate. Call `session.check()` before a run if you want
  a whole graph run to stop early.
- Tested with LangGraph 1.2.20.

## Reports

`session.report()` returns a dictionary with `report_version: 1`. Money values are
strings with eight decimal places. `None` means "not limited".

```
report_version, generated_at, currency, notes[]
agent:    name, budget{limit, max_operation_cost, actual_spending, reserved_amount,
          remaining_available, cost_by_source{reported, calculated, reconciled,
          estimated_fallback}, budget_exhausted}, usage{...}, execution{...}
session:  id, name, state, + the same budget/usage/execution blocks
enforcement_events[]: {timestamp, scope, kind, detail}
```

`usage` contains `input_tokens`, `output_tokens`, `total_tokens`,
`cached_input_tokens`, `token_limit`, `model_requests`, `model_requests_reserved`,
`model_request_limit`, `tool_executions`, `tool_executions_reserved`,
`tool_execution_limit`, `successful_operations`, `failed_operations`.
`execution` contains `elapsed_seconds`, `time_limit_seconds`,
`remaining_seconds`, `operations_seconds_total`.

Reserved amounts are estimates, not spending. The `notes` field repeats this.

## CLI

```text
owl-budget init [--path owl-budget.yaml] [--force]   create a starter file (asks before overwriting)
owl-budget validate [--config FILE]                  check a file; exit 1 on errors
owl-budget status [--config FILE]                    show configured limits and policies
owl-budget report [--config FILE] [--input REPORT.json] [--format text|json]
owl-budget pricing [--config FILE] [--format text|json]
owl-budget version
```

- `--config` defaults to `./owl-budget.yaml`.
- With a `store` configured, `status` and `report` show the stored budget. They
  open the file read-only, so they never take ownership from a running guard.
- Without a store, `status` shows the configured limits and says that state is in
  memory only.
- `report` without `--input` renders an empty report for the configuration, and
  says so. With `--input`, it renders a JSON report saved from `session.report()`.
- `python -m owl_agent_budget_guard` is equivalent to `owl-budget`.

## Budget enforcement

1. **Admission** (`reserve`, `operation`): under the lock, the guard checks the
   time limit, token exhaustion, request capacity, the operation cap, and then the
   cost against the agent and session scopes. Only if all pass is the estimate
   added to the reserved amounts of both scopes.
2. **Accounting** (`record_*`): usage is added to the counters immediately.
   Token limits are checked right after recording. The usage has already happened,
   so the guard can only report and refuse later work.
3. **Settlement**: the reserved estimate is removed and the charged amount is
   added, in one step, under the same lock. Breakdown by source is updated.

Because admission and settlement are atomic, two threads cannot both reserve the
last of the budget.

## Enforcement boundaries and limitations

Read this section before relying on the guard.

- **It does not intercept anything.** The guard enforces only the numbers your
  code reports. It does not patch provider SDKs, sit in front of an API, or see
  requests made outside your code. An API call that is not wrapped in an
  operation is not counted.
- **Costs are measured or reported by you.** Calculated costs are local estimates
  of provider billing. They are not invoices. Provider-side billing errors,
  price changes, currency conversion and taxes are not handled. There is no FX
  support: every amount must be in the configured currency.
- **Execution time is detected, not enforced.** The guard uses a monotonic clock
  and refuses new work once the limit has passed. It cannot stop Python code or
  a remote request already running. A time limit passed during an operation is
  reported after the operation settles.
- **Output tokens are checked after the call.** Use your provider's `max_tokens`
  parameter to cap output before the call.
- **Failed operations can hide cost.** If an operation fails before recording
  its usage, its cost is not charged, because the guard cannot know it. Record
  usage as soon as the provider reports it.
- **Prompt injection and untrusted tools are out of scope.** A malicious prompt or
  tool can still run up cost inside an operation that the guard admitted. Limits
  bound the damage. They do not prevent it.
- **One process per store.** A store file has one live owner at a time (see
  [Persistence](#persistence)). Several processes or machines cannot enforce one shared
  budget. Distributed enforcement is not implemented, and a network filesystem is not
  supported for the store.
- **Unencrypted file.** The store is a plain SQLite file. It holds amounts, counters,
  and event details, but no prompts, responses, or credentials. Protect the file with
  normal filesystem permissions.
- **Synchronous context managers.** `session.operation()` is a regular context
  manager, not an async one. A reservation is an ordinary object that stays open
  across `await`, and the lock is held only during each accounting call, so the
  guard can be used from async code. Async context managers are not provided yet.
- **Integrations are narrow.** The OpenAI helpers wrap one call at a time. There is
  no streaming usage accounting. The LangGraph wrapper budgets node runs, not the
  individual tool calls inside a node.
- **Default unknown-cost policy blocks.** An operation with no recorded cost
  raises `UnknownCostError` after settlement, so you will see the gap.

## Security considerations

- Configuration uses `yaml.SafeLoader` and rejects duplicate keys. No Python
  objects are built from YAML. The file size is capped at 256 KiB.
- The package makes no network requests. Pricing is never fetched.
- No API keys or secrets are read, stored or printed. Configuration has no field
  for credentials, and parse errors do not echo file contents.
- Reservation identifiers are random (`secrets.token_hex`).
- Enforcement events hold scope names, rule names and amounts. They do not hold
  prompts, responses or credentials. Do not add them yourself.
- Automatic budget increases are not implemented. `allow_budget_extension: true`
  is rejected.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests examples
ruff format --check src tests examples
pytest
pytest --cov=owl_agent_budget_guard --cov-report=term-missing
```

## Testing

The suite has 235 tests in `tests/`:

- `test_money.py`, `test_pricing.py`: decimal handling, cost arithmetic, rounding, invalid pricing.
- `test_budget.py`: reservations, settlement, policies, agent and session scopes, closing.
- `test_operations.py`: the `operation()` context, usage recording, request, token and time limits.
- `test_concurrency.py`: 32 threads racing for the same budget; exact totals afterwards.
- `test_persistence.py`: restart recovery, crash recovery, ownership takeover, failed writes, atomic saves, schema checks.
- `test_store_cli.py`: configured stores, read-only CLI views, and the rule that inspection never claims a store.
- `test_openai_integration.py`: OpenAI usage extraction with the real `openai` response types, no network.
- `test_langgraph_integration.py`: nodes run inside a real compiled LangGraph graph, sync and async.
- `test_config.py`, `test_cli.py`, `test_edges.py`, `test_report.py`, `test_isolation.py`: configuration, CLI, defensive branches, reports, and a check that the core imports no optional framework.

Time-limit tests use a fake clock and do not sleep. Last local run: 235 tests passed
with `-W error::ResourceWarning`, and 97% total coverage measured by `pytest-cov`
with branch coverage enabled. CI enforces at least 90%.

## Roadmap

1. Async context managers for `session.operation()`.
2. Streaming usage accounting for OpenAI responses.
3. Multi-process enforcement through a shared, server-side store.
4. Approval flow for the `pause` policy.

## License

MIT. See [LICENSE](LICENSE).
