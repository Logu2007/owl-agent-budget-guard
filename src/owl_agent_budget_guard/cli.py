"""Command-line interface: ``owl-budget``.

When the configuration has a ``store`` section, ``status`` and ``report`` read the
stored budget read-only. Otherwise ``report`` renders an empty report or a report
saved from ``Session.report()``, and ``status`` shows configured limits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from owl_agent_budget_guard import __version__
from owl_agent_budget_guard.config import (
    STARTER_CONFIG,
    build_guard,
    build_limits,
    load_config,
)
from owl_agent_budget_guard.exceptions import ConfigError, PersistenceError
from owl_agent_budget_guard.report import REPORT_VERSION, format_report_text
from owl_agent_budget_guard.store_report import build_store_report, format_store_report_text

DEFAULT_CONFIG_PATH = Path("owl-budget.yaml")

app = typer.Typer(
    name="owl-budget",
    help="Validate budget configuration and inspect budget reports.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = typer.Option(
    DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to the budget configuration file."
)
FormatOption = typer.Option("text", "--format", "-f", help="Output format: text or json.")


def _load_or_exit(path: Path) -> Any:
    try:
        return load_config(path)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None


def _check_format(output_format: str) -> None:
    if output_format not in ("text", "json"):
        typer.echo("Error: --format must be 'text' or 'json'", err=True)
        raise typer.Exit(code=2)


@app.command()
def init(
    path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--path", "-p", help="Where to write the file."),
    force: bool = typer.Option(False, "--force", help="Overwrite without asking."),
) -> None:
    """Create a starter configuration file."""
    if path.exists() and not force:
        if not typer.confirm(f"{path} already exists. Overwrite it?", default=False):
            typer.echo("Aborted. The existing file was not changed.")
            raise typer.Exit(code=1)
    try:
        path.write_text(STARTER_CONFIG, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Error: cannot write {path}: {exc.strerror}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Created {path}. Run 'owl-budget validate --config {path}' to check it.")


@app.command()
def validate(config: Path = ConfigOption) -> None:
    """Validate a configuration file and report every error found."""
    loaded = _load_or_exit(config)
    typer.echo(
        f"OK: {config} is valid "
        f"(agent={loaded.agent.name}, currency={loaded.budget.currency}, "
        f"max_cost={loaded.budget.max_cost}, pricing_models={len(loaded.pricing.models)})"
    )


@app.command()
def status(config: Path = ConfigOption) -> None:
    """Show the configured limits, and the stored budget state when a store is configured.

    The stored view is read-only, so it never interferes with a running guard.
    """
    loaded = _load_or_exit(config)
    limits = loaded.limits
    typer.echo(f"Agent:              {loaded.agent.name}")
    typer.echo(f"Budget (agent):     {loaded.budget.max_cost} {loaded.budget.currency}")
    typer.echo(f"Max model requests: {limits.max_llm_calls}")
    typer.echo(f"Max tool calls:     {limits.max_tool_calls}")
    typer.echo(f"Max total tokens:   {limits.max_tokens}")
    typer.echo(f"Max execution time: {limits.max_execution_time} s")
    typer.echo(f"On budget exceeded: {loaded.policies.on_budget_exceeded}")
    typer.echo(f"Unknown cost:       {loaded.policies.unknown_cost}")
    if loaded.store is None:
        typer.echo("")
        typer.echo(
            "No store is configured, so budget state is in memory only. Add a 'store' "
            "section to the configuration to persist it."
        )
        return

    report = _store_report_or_exit(loaded)
    agent = report["agent"]["budget"]
    typer.echo("")
    typer.echo(f"Store:              {loaded.store.path}")
    typer.echo(f"Actual spending:    {agent['actual_spending']} {loaded.budget.currency}")
    typer.echo(f"Reserved (estimate): {agent['reserved_amount']} {loaded.budget.currency}")
    remaining = agent["remaining_available"]
    typer.echo(
        f"Remaining available: {'unlimited' if remaining is None else remaining} "
        f"{loaded.budget.currency if remaining is not None else ''}".rstrip()
    )
    typer.echo(f"Sessions stored:    {len(report['sessions'])}")


def _store_report_or_exit(loaded: Any) -> dict[str, Any]:
    try:
        return build_store_report(
            loaded.store.path,
            agent_name=loaded.agent.name,
            currency=loaded.budget.currency,
            limits=build_limits(loaded),
        )
    except PersistenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None


def _load_report_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        typer.echo(f"Error: cannot read {path}: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: {path} is not valid JSON (line {exc.lineno})", err=True)
        raise typer.Exit(code=1) from None
    if (
        not isinstance(data, dict)
        or data.get("report_version") != REPORT_VERSION
        or "agent" not in data
        or "session" not in data
    ):
        typer.echo(
            f"Error: {path} is not an OWL budget report (expected report_version "
            f"{REPORT_VERSION} with 'agent' and 'session' sections)",
            err=True,
        )
        raise typer.Exit(code=1)
    return data


@app.command()
def report(
    config: Path = ConfigOption,
    input_file: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Render a report saved as JSON from Session.report().",
    ),
    output_format: str = FormatOption,
) -> None:
    """Render a usage report, from a saved file or an empty report for the configuration."""
    _check_format(output_format)
    loaded = _load_or_exit(config)
    if input_file is not None:
        data = _load_report_file(input_file)
    elif loaded.store is not None:
        # Read-only: building a guard here would claim the store from a running process.
        data = _store_report_or_exit(loaded)
        if output_format == "json":
            typer.echo(json.dumps(data, indent=2))
        else:
            typer.echo(format_store_report_text(data))
        return
    else:
        guard = build_guard(loaded)
        with guard.session("cli-report") as session:
            data = session.report()
        data["note"] = (
            "No usage was recorded: this report was generated by the CLI for the "
            "configuration alone. Use --input with a report saved from Session.report()."
        )
    if output_format == "json":
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(format_report_text(data))


@app.command()
def pricing(config: Path = ConfigOption, output_format: str = FormatOption) -> None:
    """List the model prices registered in the configuration."""
    _check_format(output_format)
    loaded = _load_or_exit(config)
    entries = [
        {
            "model": name,
            "currency": entry.currency,
            "input_per_million": str(entry.input_per_million),
            "output_per_million": str(entry.output_per_million),
            "cached_input_per_million": (
                None
                if entry.cached_input_per_million is None
                else str(entry.cached_input_per_million)
            ),
            "source": entry.source,
        }
        for name, entry in sorted(loaded.pricing.models.items())
    ]
    if output_format == "json":
        typer.echo(json.dumps({"models": entries}, indent=2))
        return
    if not entries:
        typer.echo(
            "No pricing configured. No prices are bundled with OWL; add verified prices under "
            "pricing.models in the configuration file."
        )
        return
    header = ("MODEL", "CURRENCY", "INPUT/1M", "OUTPUT/1M", "CACHED/1M", "SOURCE")
    rows = [
        (
            e["model"],
            e["currency"],
            e["input_per_million"],
            e["output_per_million"],
            e["cached_input_per_million"] or "-",
            e["source"] or "-",
        )
        for e in entries
    ]
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        typer.echo("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


@app.command()
def version() -> None:
    """Show the package version."""
    typer.echo(f"owl-budget {__version__}")


def main() -> None:
    app(prog_name="owl-budget")


if __name__ == "__main__":
    main()
