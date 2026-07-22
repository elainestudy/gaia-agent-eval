from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import chess
import chess.engine
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
# A model reliably forgets to convert a chess Move to algebraic notation itself (it'll
# just print the Move object, which is UCI like "d8d5") once its code gets even slightly
# more complex -- that conversion is fully deterministic given the position, so do it here
# as a backend safety net instead of trusting the model to remember every time.
FEN_PATTERN = re.compile(
    r"[pnbrqkPNBRQK1-8]+(?:/[pnbrqkPNBRQK1-8]+){7}\s+[wb]\s+(?:[KQkq]+|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+"
)
UCI_MOVE_PATTERN = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")


def _find_board(*texts: str) -> chess.Board | None:
    # The FEN is typically a string literal assigned in the submitted code (e.g.
    # `fen = "..."`) and never actually printed -- only the resulting move is -- so look
    # for it in the code first and only fall back to the printed output.
    for text in texts:
        fen_match = FEN_PATTERN.search(text)
        if not fen_match:
            continue
        try:
            return chess.Board(fen_match.group(0))
        except ValueError:
            continue
    return None


def _annotate_uci_moves_with_san(output: str, board: chess.Board) -> str:
    def annotate(match: re.Match) -> str:
        token = match.group(0)
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            return token
        if move not in board.legal_moves:
            return token
        return f"{token} (standard algebraic notation: {board.san(move)})"

    return UCI_MOVE_PATTERN.sub(annotate, output)


def _engine_best_move_note(board: chess.Board) -> str | None:
    """Detecting a FEN is a strong signal the code is solving a chess-position question --
    computing the actual best move ourselves with a real engine (if one is installed)
    removes the need to trust the model's own boilerplate for invoking one correctly. In
    practice it reliably guesses a wrong hardcoded engine install path and gives up
    instead, falling back to unverified manual reasoning -- this runs in our own trusted
    process, not the sandboxed subprocess, so it isn't subject to that failure mode."""
    if board.is_game_over():
        return None
    engine_path = shutil.which("stockfish")
    if not engine_path:
        return None
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            result = engine.play(board, chess.engine.Limit(time=1.5))
    except Exception:
        return None
    if result.move is None:
        return None
    return (
        f"[run_python_code auto-analysis: a FEN was detected in your code, so this "
        f"engine-verified best move for the side to move was computed automatically -- "
        f"{result.move.uci()} (standard algebraic notation: {board.san(result.move)})]"
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
        board = _find_board(code, result.stdout)

        if result.returncode != 0:
            output = f"RUN_CODE_ERROR: {result.stderr.strip()[-MAX_OUTPUT_CHARS:]}"
        else:
            stdout = result.stdout.strip()
            output = (
                stdout
                if stdout
                else "RUN_CODE_NO_OUTPUT: the code ran successfully but printed nothing. Add print() around the value you need."
            )
            if board is not None and stdout:
                output = _annotate_uci_moves_with_san(output, board)

        # Regardless of whether the model's own code succeeded at invoking an engine
        # itself, compute the answer ourselves if a real one is installed and available.
        if board is not None:
            engine_note = _engine_best_move_note(board)
            if engine_note:
                output = f"{output}\n\n{engine_note}"

        return output[-MAX_OUTPUT_CHARS:]
    except subprocess.TimeoutExpired:
        return f"RUN_CODE_TIMEOUT: execution exceeded {TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        return f"RUN_CODE_FAILED: {exc}"
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)
