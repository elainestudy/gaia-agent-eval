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


def _format_score(score: chess.engine.PovScore, turn: chess.Color) -> str:
    relative = score.pov(turn)
    mate = relative.mate()
    if mate is not None:
        return f"mate in {abs(mate)}"
    cp = relative.score()
    return "n/a" if cp is None else f"{cp / 100:+.2f}"


def _engine_candidate_moves_note(board: chess.Board, multipv: int = 3) -> str | None:
    """Detecting a FEN is a strong signal the code is solving a chess-position question --
    but computing a single "best" move ourselves and handing it over as the answer lets a
    real engine's raw evaluation silently override the model's own judgment. When a
    position has more than one objectively winning move (common in "find the move that
    wins" puzzles), the highest-eval move is not necessarily the one a puzzle's answer key
    expects -- so surface the engine's top candidates with their evaluation and likely
    follow-up instead, and leave picking one to the model."""
    if board.is_game_over():
        return None
    engine_path = shutil.which("stockfish")
    if not engine_path:
        return None
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            infos = engine.analyse(board, chess.engine.Limit(time=1.5), multipv=multipv)
    except Exception:
        return None

    lines = []
    for entry in infos:
        pv = entry.get("pv")
        score = entry.get("score")
        if not pv or score is None:
            continue
        preview_board = board.copy()
        sans = []
        for move in pv[:4]:
            sans.append(preview_board.san(move))
            preview_board.push(move)
        lines.append(f"{sans[0]} (score: {_format_score(score, board.turn)}, likely continuation: {' '.join(sans)})")
    if not lines:
        return None

    side_to_move = "white" if board.turn else "black"
    candidates = "\n".join(f"  - {line}" for line in lines)
    return (
        f"[run_python_code auto-analysis: a FEN was detected in your code ({side_to_move} "
        f"to move per that FEN -- confirm this matches the question/image). A real chess "
        f"engine's top {len(lines)} legal candidate moves, highest-scoring first:\n{candidates}\n"
        "Score is the engine's raw evaluation, not a verdict: if more than one candidate is "
        "clearly winning, the highest score is not automatically the answer the question "
        "wants -- judge which candidate best fits the question yourself rather than "
        "defaulting to the top score.]"
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
            output = f"RUN_CODE_ERROR: {result.stderr.strip()[-MAX_OUTPUT_CHARS:]}"
        else:
            stdout = result.stdout.strip()
            output = (
                stdout
                if stdout
                else "RUN_CODE_NO_OUTPUT: the code ran successfully but printed nothing. Add print() around the value you need."
            )
            board = _find_board(code, result.stdout)
            if board is not None and stdout:
                output = _annotate_uci_moves_with_san(output, board)

            # Regardless of whether the model's own code succeeded at invoking an engine
            # itself, surface candidate moves ourselves if a real one is installed and
            # available -- as information to weigh, not a chosen answer.
            if board is not None:
                engine_note = _engine_candidate_moves_note(board)
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
