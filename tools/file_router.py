import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Single-image attachments go through Gemini's own multimodal input instead of the local
# Ollama model used elsewhere (tools/video_local.py) -- empirically far more reliable for
# precise structured reading (e.g. a chessboard's actual coordinate orientation). A GAIA
# task has at most one image attachment, so the added API cost per question is bounded;
# a video can produce many sampled frames, so that path stays on the free local model.
GEMINI_VISION_MODEL = "gemini-3.1-flash-lite"
_gemini_vision_llm: ChatGoogleGenerativeAI | None = None

ATTACHMENT_CACHE_DIR = Path(".cache/attachments")
# agents-course-unit4-scoring.hf.space/files/{task_id} 404s for tasks that do have a
# file (known upstream bug: https://discuss.huggingface.co/t/get-files-task-id-returning-404-for-every-task-id-with-file-name/170575).
# When that happens, fall back to fetching the file directly from the dataset on the
# Hub and cache it here instead, so re-runs don't need the Hub at all.
LOCAL_ATTACHMENTS_DIR = Path(".local_attachments")
GAIA_DATASET_REPO = "gaia-benchmark/GAIA"
# The course's task set draws from the validation split (it needs public ground truth
# to score), but try test too in case a future task set includes one of those files.
GAIA_DATASET_SPLIT_PREFIXES = ("2023/validation", "2023/test")
VideoLogger = Callable[[str], None]

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".html", ".htm", ".log"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls"}
WHISPER_MODEL_SIZE = "base"

AMBIGUITY_HINTS = (
    (r"\bbird\b", r"\bpenguin(s)?\b", "Penguins count as birds for bird-species questions."),
    (r"\bdrink\b", r"\bsoda\b", "Soda is a drink."),
    (r"\binput device\b", r"\bkeyboard\b", "Keyboard is an input device."),
)


def ensure_attachment_cache_dir(task_id: str) -> Path:
    task_dir = ATTACHMENT_CACHE_DIR / re.sub(r"[^0-9A-Za-z_-]+", "_", task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def guess_extension_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ".bin"
    main_type = content_type.split(";", 1)[0].strip().lower()
    guessed = mimetypes.guess_extension(main_type) or ""
    if guessed == ".jpe":
        return ".jpg"
    return guessed or ".bin"


def detect_file_type(file_path: Path, content_type: str | None = None) -> str:
    suffix = file_path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if suffix == ".pdf":
        return "pdf"

    try:
        sample = file_path.read_bytes()[:512]
    except Exception:
        return "unknown"

    if sample.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if sample.startswith(b"\xff\xd8\xff"):
        return "image"
    if sample.startswith(b"GIF87a") or sample.startswith(b"GIF89a"):
        return "image"
    if len(sample) >= 12 and sample.startswith(b"RIFF") and sample[8:12] == b"WEBP":
        return "image"
    if sample.startswith(b"BM"):
        return "image"
    if sample[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image"

    if b"\x00" in sample:
        return "binary"

    if sample.startswith(b"%PDF"):
        return "pdf"

    if content_type:
        main_type = content_type.split(";", 1)[0].strip().lower()
        if main_type.startswith("text/"):
            return "text"
        if main_type.startswith("image/"):
            return "image"
        if main_type.startswith("video/"):
            return "video"

    try:
        sample.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "binary"


def save_attachment_response(task_id: str, response: requests.Response) -> Path:
    output_dir = ensure_attachment_cache_dir(task_id)
    content_type = response.headers.get("content-type", "").lower()
    extension = guess_extension_from_content_type(content_type)
    if "application/json" in content_type:
        file_path = output_dir / f"{task_id}.json"
        file_path.write_text(json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf-8")
        return file_path

    file_path = output_dir / f"{task_id}{extension}"
    if content_type.startswith("text/") or extension in {".txt", ".md", ".csv", ".tsv", ".xml", ".html", ".htm", ".log"}:
        file_path.write_text(response.text, encoding="utf-8")
    else:
        file_path.write_bytes(response.content)
    return file_path


def _local_attachment_cache_path(task_id: str, file_name: str) -> Path:
    task_dir = LOCAL_ATTACHMENTS_DIR / re.sub(r"[^0-9A-Za-z_-]+", "_", task_id)
    # file_name comes from GAIA task metadata; take only the basename so a value like
    # "../../evil.txt" can't resolve the cache path outside task_dir.
    return task_dir / Path(file_name).name


def _download_from_hub_fallback(task_id: str, file_name: str, emit: VideoLogger | None = None) -> Path | None:
    token = os.getenv("HF_TOKEN")
    downloaded_path = None
    last_error: Exception | None = None
    for split_prefix in GAIA_DATASET_SPLIT_PREFIXES:
        try:
            downloaded_path = hf_hub_download(
                repo_id=GAIA_DATASET_REPO,
                filename=f"{split_prefix}/{file_name}",
                repo_type="dataset",
                token=token,
            )
            break
        except EntryNotFoundError as exc:
            # Expected: file genuinely isn't in this split, try the next one.
            last_error = exc
        except (GatedRepoError, HfHubHTTPError) as exc:
            # Auth/access/network problem, not a "file not found" -- same token and repo
            # for every split, so retrying the next split would fail identically. Surface
            # this distinctly instead of masking it as a generic miss.
            if emit:
                emit(
                    f"[file-router] Hub fallback auth/access error for {file_name} "
                    f"(check HF_TOKEN and gated-dataset access): {exc}"
                )
            return None
        except Exception as exc:
            last_error = exc

    if downloaded_path is None:
        if emit:
            emit(f"[file-router] Hub fallback failed for {file_name}: {last_error}")
        return None

    dest_file = _local_attachment_cache_path(task_id, file_name)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_bytes(Path(downloaded_path).read_bytes())
    if emit:
        emit(f"[file-router] downloaded attachment via Hub fallback: {dest_file}")
    return dest_file


def download_attachment(
    task_id: str,
    api_base: str,
    emit: VideoLogger | None = None,
    file_name: str | None = None,
) -> tuple[Path | None, str | None]:
    if file_name:
        cached_path = _local_attachment_cache_path(task_id, file_name)
        if cached_path.exists():
            if emit:
                emit(f"[file-router] using locally cached attachment: {cached_path}")
            return cached_path, mimetypes.guess_type(str(cached_path))[0]

    response = requests.get(f"{api_base}/files/{task_id}", timeout=120)
    if response.status_code == 404:
        if file_name:
            fallback_path = _download_from_hub_fallback(task_id, file_name, emit=emit)
            if fallback_path:
                return fallback_path, mimetypes.guess_type(str(fallback_path))[0]
        return None, None

    response.raise_for_status()
    file_path = save_attachment_response(task_id, response)
    content_type = response.headers.get("content-type", "").lower()
    if emit:
        emit(f"[file-router] downloaded attachment: {file_path} ({content_type or 'unknown content-type'})")
    return file_path, content_type


def parse_textual_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        try:
            parsed = json.loads(file_path.read_text(encoding="utf-8"))
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix in {".csv", ".tsv", ".txt", ".md", ".yaml", ".yml", ".xml", ".html", ".htm", ".log"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        if raw_text.startswith("%PDF"):
            return "UNSUPPORTED_TEXT_EXTRACTION: PDF detected but no local text extractor is installed."
        return raw_text
    except UnicodeDecodeError:
        return "UNSUPPORTED_TEXT_EXTRACTION: file is not readable as UTF-8 text."


VISION_PROMPT = (
    "You are looking at one visual attachment from a GAIA task.\n"
    "Describe the visible content as thoroughly and systematically as possible: every "
    "distinct object, symbol, or piece of text visible, and its position or spatial "
    "relationship to the others. Do not stop after the first few salient items.\n"
    "If the image shows a grid, board, table, map, or other structured layout with its own "
    "coordinate labels (e.g. a chessboard with rank numbers and file letters): first read "
    "and state the coordinate labels themselves, in the order they actually appear in the "
    "image -- do not assume a standard orientation, since labels can be reversed, rotated, "
    "or mirrored. Then scan systematically row by row and describe every occupied cell "
    "using those exact labels; do not skip any occupied cell.\n"
    "For a chessboard specifically: after the row-by-row scan, also list every occupied "
    "square on its own line in exactly this format (one line per piece, using the square's "
    "real label as read from the image, not an assumed one):\n"
    "BOARD_SQUARE: <square> <white|black> <king|queen|rook|bishop|knight|pawn>\n"
    "Before finalizing, count the BOARD_SQUARE lines and confirm there is exactly one white "
    "king and exactly one black king among them; if not, look again and correct it.\n"
    "Do not guess the final answer or interpret what the content means; only report what is "
    "visible, exhaustively and precisely.\n"
    "If an object is a subclass of a broader concept, mention the broader concept too.\n"
    "Examples: penguin -> bird, soda -> drink, keyboard -> input device.\n"
    "Return plain text, not JSON."
)
BOARD_SQUARE_LINE_PATTERN = re.compile(r"BOARD_SQUARE\w*:\s*(.+)", re.IGNORECASE)
SQUARE_TOKEN_PATTERN = re.compile(r"\b([a-h][1-8])\b", re.IGNORECASE)
COLOR_TOKEN_PATTERN = re.compile(r"\b(white|black)\b", re.IGNORECASE)
PIECE_TOKEN_PATTERN = re.compile(r"\b(king|queen|rook|bishop|knight|pawn)\b", re.IGNORECASE)
PIECE_LETTERS = {"king": "k", "queen": "q", "rook": "r", "bishop": "b", "knight": "n", "pawn": "p"}
CHESS_CONTENT_PATTERN = re.compile(r"\bchess\b", re.IGNORECASE)


def _parse_board_squares(text: str) -> list[tuple[str, str, str]] | None:
    # Look for the three required tokens anywhere on the line rather than requiring an
    # exact literal format -- the model reliably names square/color/piece but not
    # reliably the punctuation around them (brackets, colons, commas all show up).
    results: list[tuple[str, str, str]] = []
    for line_match in BOARD_SQUARE_LINE_PATTERN.finditer(text):
        rest = line_match.group(1)
        square_match = SQUARE_TOKEN_PATTERN.search(rest)
        color_match = COLOR_TOKEN_PATTERN.search(rest)
        piece_match = PIECE_TOKEN_PATTERN.search(rest)
        if square_match and color_match and piece_match:
            results.append((square_match.group(1).lower(), color_match.group(1).lower(), piece_match.group(1).lower()))
    return results or None


def _build_validated_board_fen(squares: list[tuple[str, str, str]]) -> str | None:
    """Build a board from the model's per-square readings ourselves rather than trusting
    the model to assemble a correct FEN string itself -- and reject it outright if it's
    not even a legal-looking chess position (missing/duplicate king), since that's a
    reliable, cheap signal that the reading itself was wrong."""
    import chess

    board = chess.Board(None)
    seen_squares: set[str] = set()
    try:
        for square_name, color, piece in squares:
            if square_name in seen_squares:
                return None
            seen_squares.add(square_name)
            symbol = PIECE_LETTERS[piece]
            if color == "white":
                symbol = symbol.upper()
            board.set_piece_at(chess.parse_square(square_name), chess.Piece.from_symbol(symbol))
    except ValueError:
        return None

    if len(board.pieces(chess.KING, chess.WHITE)) != 1 or len(board.pieces(chess.KING, chess.BLACK)) != 1:
        return None
    return board.board_fen()


def _get_gemini_vision_llm() -> ChatGoogleGenerativeAI:
    global _gemini_vision_llm
    if _gemini_vision_llm is None:
        _gemini_vision_llm = ChatGoogleGenerativeAI(model=GEMINI_VISION_MODEL)
    return _gemini_vision_llm


def _normalize_gemini_vision_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()


def gemini_vision_describe(prompt: str, image_path: Path, emit: VideoLogger | None = None) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ]
    )
    if emit:
        emit(f"[file-router] calling Gemini vision model={GEMINI_VISION_MODEL} on {image_path.name}")
    response = _get_gemini_vision_llm().invoke([message])
    return _normalize_gemini_vision_content(response.content)


def ocr_or_vision_for_visual_content(
    file_path: Path,
    emit: VideoLogger | None = None,
) -> str:
    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return f"VISUAL_ANALYSIS_UNAVAILABLE: unsupported visual file type {file_path.suffix or '<none>'}."

    description = gemini_vision_describe(VISION_PROMPT, file_path, emit=emit)

    squares = _parse_board_squares(description)
    board_fen = _build_validated_board_fen(squares) if squares else None

    # Retry once whenever this looks like a chessboard but we didn't come away with a
    # valid FEN -- whether because the model ignored the BOARD_SQUARE format entirely or
    # because the squares it did list didn't form a legal-looking position.
    if not board_fen and CHESS_CONTENT_PATTERN.search(description):
        if emit:
            emit("[file-router] no valid board FEN parsed from a likely chessboard image, retrying vision call once")
        retry_prompt = VISION_PROMPT + (
            "\n\nYour previous attempt did not produce a valid BOARD_SQUARE line for every "
            "occupied square (or the resulting position was invalid, e.g. a missing or "
            "duplicate king) -- look at the image again more carefully and output one "
            "BOARD_SQUARE line per occupied square, exactly in the format instructed above."
        )
        description = gemini_vision_describe(retry_prompt, file_path, emit=emit)
        squares = _parse_board_squares(description)
        board_fen = _build_validated_board_fen(squares) if squares else None

    if board_fen:
        description += (
            f"\n\nBoard FEN (piece placement only, parsed from the BOARD_SQUARE lines "
            f"above and validated with python-chess -- exactly one king per side): {board_fen}"
        )
    return description


_whisper_model_cache: dict[str, Any] = {}


def _get_whisper_model(model_size: str = WHISPER_MODEL_SIZE) -> Any:
    if model_size not in _whisper_model_cache:
        from faster_whisper import WhisperModel

        _whisper_model_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model_cache[model_size]


def transcribe_audio_content(file_path: Path, emit: VideoLogger | None = None) -> str:
    if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
        return f"AUDIO_TRANSCRIPTION_UNAVAILABLE: unsupported audio file type {file_path.suffix or '<none>'}."

    try:
        model = _get_whisper_model()
        if emit:
            emit(f"[file-router] transcribing {file_path.name} with local Whisper ({WHISPER_MODEL_SIZE})")
        segments, _info = model.transcribe(str(file_path))
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        return f"UNSUPPORTED_AUDIO_EXTRACTION: failed to transcribe {file_path.name}: {exc}"

    if not text:
        return "NO_TRANSCRIPT_AVAILABLE: audio transcription produced no text."
    return text


def parse_spreadsheet_content(file_path: Path) -> str:
    if file_path.suffix.lower() not in SPREADSHEET_EXTENSIONS:
        return f"SPREADSHEET_PARSE_UNAVAILABLE: unsupported spreadsheet file type {file_path.suffix or '<none>'}."

    try:
        import pandas as pd

        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as exc:
        return f"UNSUPPORTED_SPREADSHEET_EXTRACTION: failed to parse {file_path.name}: {exc}"

    parts: list[str] = []
    for sheet_name, dataframe in sheets.items():
        parts.append(f"Sheet: {sheet_name}")
        parts.append(dataframe.to_csv(index=False))
    return "\n\n".join(parts).strip()


def normalize_entities(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines()]
    collapsed_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                collapsed_lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        collapsed_lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(collapsed_lines).strip()


def knowledge_lookup_if_ambiguous(question_text: str, evidence_text: str) -> str:
    lowered_question = question_text.lower()
    lowered_evidence = evidence_text.lower()

    hint_lines: list[str] = []
    for concept_pattern, entity_pattern, hint in AMBIGUITY_HINTS:
        if re.search(concept_pattern, lowered_question) and re.search(entity_pattern, lowered_evidence):
            hint_lines.append(f"- {hint}")

    ambiguity_signals = ("uncertain", "maybe", "possibly", "appears to be", "looks like", "not sure", "ambiguous")
    if any(signal in lowered_evidence for signal in ambiguity_signals):
        hint_lines.append("- Evidence is ambiguous; search for a confirming knowledge source before finalizing.")

    if not hint_lines:
        return "Knowledge lookup: not required yet."

    return "Knowledge lookup recommended:\n" + "\n".join(hint_lines)


def build_attachment_evidence(
    api_base: str,
    task_id: str,
    question_text: str,
    emit: VideoLogger | None = None,
    file_name: str | None = None,
) -> str:
    attachment_path, content_type = download_attachment(
        task_id=task_id, api_base=api_base, emit=emit, file_name=file_name
    )
    if attachment_path is None:
        return "Attachment evidence: none."

    file_type = detect_file_type(attachment_path, content_type=content_type)
    if emit:
        emit(f"[file-router] detected file type: {file_type}")

    if file_type == "text":
        raw_content = parse_textual_content(attachment_path)
    elif file_type == "image":
        raw_content = ocr_or_vision_for_visual_content(attachment_path, emit=emit)
    elif file_type == "audio":
        raw_content = transcribe_audio_content(attachment_path, emit=emit)
    elif file_type == "spreadsheet":
        raw_content = parse_spreadsheet_content(attachment_path)
    elif file_type == "pdf":
        raw_content = "UNSUPPORTED_FILE_TYPE: PDF attachment detected, but no local PDF text/vision extractor is configured."
    else:
        raw_content = f"UNSUPPORTED_FILE_TYPE: {attachment_path.name} ({file_type})."

    normalized_content = normalize_entities(raw_content)
    lookup_hint = knowledge_lookup_if_ambiguous(question_text=question_text, evidence_text=normalized_content)

    evidence_lines = [
        "Attachment evidence summary:",
        f"- file: {attachment_path.name}",
        f"- detected type: {file_type}",
        f"- parsed/observed content: {normalized_content}",
        f"- normalization: entities and whitespace normalized",
        f"- {lookup_hint}",
    ]
    return "\n".join(evidence_lines)
