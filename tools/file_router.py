import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Callable

import requests
from huggingface_hub import hf_hub_download

from tools.video_local import DEFAULT_VISION_MODEL, ollama_chat_with_image

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


def _local_attachment_cache_path(task_id: str, file_name: str) -> Path:
    task_dir = LOCAL_ATTACHMENTS_DIR / re.sub(r"[^0-9A-Za-z_-]+", "_", task_id)
    return task_dir / file_name


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
