from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 4000
# This tool is scoped to pure computation (aggregating/summing, evaluating an attached
# script's output). Anything reaching outside that -- filesystem, network, process,
# environment access -- is out of scope, so reject it up front instead of executing it.
DISALLOWED_PATTERN = re.compile(
    r"\b(?:import\s+(?:os|sys|subprocess|socket|shutil|requests|urllib|http|ftplib|"
    r"smtplib|ctypes|pty|multiprocessing)\b|"
    r"__import__\s*\(|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\bcompile\s*\()"
)


@tool
def run_python_code(code: str) -> str:
    """Execute a snippet of Python code in a real interpreter and return what it prints.
    Use this instead of mental math or wolframalpha_query whenever a query has many terms,
    involves aggregating/summing a list of numbers, or is an attached Python script whose
    actual output needs to be determined -- do not simulate execution in your head.
    The code must print() whatever value you need; only stdout is returned.
    This tool is for computation only: no file, network, subprocess, or OS access."""
    if DISALLOWED_PATTERN.search(code):
        return (
            "RUN_CODE_BLOCKED: this tool only runs pure computation (no file, network, "
            "subprocess, or OS access). Rewrite the code without imports like os/sys/"
            "subprocess/socket/shutil/requests or calls like open()/eval()/exec()."
        )

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            script_path = Path(handle.name)

        result = subprocess.run(
            [sys.executable, str(script_path)],
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
