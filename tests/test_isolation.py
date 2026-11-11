"""The core package must work without the optional framework packages installed."""

import subprocess
import sys


def test_core_import_does_not_load_optional_frameworks() -> None:
    code = (
        "import sys\n"
        "import owl_agent_budget_guard, owl_agent_budget_guard.store, "
        "owl_agent_budget_guard.cli, owl_agent_budget_guard.store_report\n"
        "loaded = [m for m in ('openai', 'langgraph') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"core imported optional frameworks: {result.stdout.strip()}"
