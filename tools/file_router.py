import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable

import requests

from tools.video_local import DEFAULT_VISION_MODEL, ollama_chat_with_image

ATTACHMENT_CACHE_DIR = Path(".cache/attachments")
VideoLogger = Callable[[str], None]

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".html", ".htm", ".log"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

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


def download_attachment(task_id: str, api_base: str, emit: VideoLogger | None = None) -> tuple[Path | None, str | None]:
    response = requests.get(f"{api_base}/files/{task_id}", timeout=120)
    if response.status_code == 404:
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


def ocr_or_vision_for_visual_content(
    file_path: Path,
    emit: VideoLogger | None = None,
    model: str = DEFAULT_VISION_MODEL,
) -> str:
    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return f"VISUAL_ANALYSIS_UNAVAILABLE: unsupported visual file type {file_path.suffix or '<none>'}."

    prompt = (
        "You are looking at one visual attachment from a GAIA task.\n"
        "Describe the visible content objectively and conservatively.\n"
        "List what is present, what it resembles, and any category ambiguities.\n"
        "Do not guess the final answer.\n"
        "If the image seems to show a subclass of a broader concept, mention the broader concept too.\n"
        "Examples: penguin -> bird, soda -> drink, keyboard -> input device.\n"
        "Return plain text, not JSON."
    )
    if emit:
        emit(f"[file-router] running local vision model on {file_path.name}")
    return ollama_chat_with_image(model=model, prompt=prompt, image_path=file_path, emit=emit)


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
) -> str:
    attachment_path, content_type = download_attachment(task_id=task_id, api_base=api_base, emit=emit)
    if attachment_path is None:
        return "Attachment evidence: none."

    file_type = detect_file_type(attachment_path, content_type=content_type)
    if emit:
        emit(f"[file-router] detected file type: {file_type}")

    if file_type == "text":
        raw_content = parse_textual_content(attachment_path)
    elif file_type == "image":
        raw_content = ocr_or_vision_for_visual_content(attachment_path, emit=emit)
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
