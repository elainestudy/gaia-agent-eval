from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool

TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 4000


@tool
def run_python_code(code: str) -> str:
    """Execute a snippet of Python code in a real interpreter and return what it prints.
    Use this instead of mental math or wolframalpha_query whenever a query has many terms,
    involves aggregating/summing a list of numbers, or is an attached Python script whose
    actual output needs to be determined -- do not simulate execution in your head.
    The code must print() whatever value you need; only stdout is returned."""
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            script_path = Path(handle.name)

        result = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return f"RUN_CODE_ERROR: {result.stderr.strip()[-MAX_OUTPUT_CHARS:]}"

        output = result.stdout.strip()
        if not output:
            return "RUN_CODE_NO_OUTPUT: the code ran successfully but printed nothing. Add print() around the value you need."
        return output[-MAX_OUTPUT_CHARS:]
    except subprocess.TimeoutExpired:
        return f"RUN_CODE_TIMEOUT: execution exceeded {TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        return f"RUN_CODE_FAILED: {exc}"
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)
