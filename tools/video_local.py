import base64
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import requests

VIDEO_CACHE_DIR = Path(".cache/video")
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_VISION_MODEL = "minicpm-v:8b"
VideoLogger = Callable[[str], None]
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
UNCERTAINTY_MARKERS = {
    "unknown",
    "unclear",
    "indeterminate",
    "possible",
    "possibly",
    "likely",
    "maybe",
    "might",
    "could be",
    "or",
    "similar to",
    "appears to be",
    "not sure",
    "can't tell",
    "cannot tell",
}


def extract_youtube_url(text: str) -> str | None:
    url_match = re.search(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=[^\s\]\)>,]+|youtu\.be/[^\s\]\)>,]+)", text)
    if not url_match:
        return None
    return url_match.group(0).rstrip(".,;:)>]\"'")


def sanitize_task_id(task_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", task_id)


def ensure_cache_dir(task_id: str) -> Path:
    task_dir = VIDEO_CACHE_DIR / sanitize_task_id(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def download_low_quality_youtube_video(url: str, task_id: str, emit: VideoLogger | None = None) -> Path:
    yt_dlp_path = shutil.which("yt-dlp")
    if yt_dlp_path is None:
        raise RuntimeError("yt-dlp is not installed or not on PATH. Install it first to download YouTube videos.")

    output_dir = ensure_cache_dir(task_id)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    command = [
        yt_dlp_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-f",
        "worstvideo",
        "-o",
        output_template,
        url,
    ]
    if emit:
        emit(f"[local-video] downloading low-quality video with yt-dlp: {url}")
    subprocess.run(command, check=True, capture_output=True, text=True)

    candidates = [
        path
        for path in output_dir.glob("*")
        if path.is_file() and path.suffix.lower() not in {".json", ".part"}
    ]
    if not candidates:
        raise RuntimeError(f"yt-dlp completed but no file was created in {output_dir}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if emit:
        emit(f"[local-video] download complete: {candidates[0]}")
    return candidates[0]


def sample_video_frames(
    video_path: Path,
    task_id: str,
    frame_interval_seconds: int = 1,
    max_frames: int = 0,
    emit: VideoLogger | None = None,
) -> list[Path]:
    frame_dir = ensure_cache_dir(task_id) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(frame_dir / "frame_%04d.jpg")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{frame_interval_seconds},scale=640:-1",
        "-q:v",
        "3",
        frame_pattern,
    ]
    if emit:
        frame_cap_label = "no hard cap" if max_frames <= 0 else f"up to {max_frames} frames"
        emit(f"[local-video] extracting frames every {frame_interval_seconds}s ({frame_cap_label})")
    subprocess.run(command, check=True, capture_output=True, text=True)

    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if emit:
        emit(f"[local-video] extracted {len(frame_paths)} frames")
    return frame_paths


def _encode_image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _extract_field_value(text: str, field_name: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.*)$", text)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_count_from_text(text: str) -> int | None:
    raw_value = _extract_field_value(text, "count_guess")
    if not raw_value:
        return None

    match = re.search(r"\b(\d+)\b", raw_value)
    if match:
        return int(match.group(1))

    lowered_value = raw_value.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered_value):
            return value

    return None


def _extract_confidence_from_text(text: str) -> str:
    raw_value = _extract_field_value(text, "confidence").lower()
    if raw_value.startswith("high"):
        return "high"
    if raw_value.startswith("medium"):
        return "medium"
    if raw_value.startswith("low"):
        return "low"
    return "unknown"


def _extract_species_candidates(text: str) -> list[str]:
    raw_value = _extract_field_value(text, "species_guess")
    if not raw_value:
        return []

    lowered_value = raw_value.lower()
    if any(marker in lowered_value for marker in UNCERTAINTY_MARKERS):
        return []

    bracketed = [match.strip() for match in re.findall(r"\[(.*?)\]", raw_value) if match.strip()]
    if bracketed:
        return bracketed

    if any(symbol in raw_value for symbol in ("?", "<", ">", "{", "}", "(", ")")):
        return []

    candidates = [item.strip() for item in re.split(r",|/|\band\b", raw_value, flags=re.IGNORECASE) if item.strip()]
    if len(candidates) > 4:
        return []

    return candidates


def ollama_chat_with_image(model: str, prompt: str, image_path: Path, emit: VideoLogger | None = None) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [_encode_image_to_base64(image_path)],
            }
        ],
        "stream": False,
    }
    if emit:
        emit(f"[local-video] calling Ollama model={model} on {image_path.name}")
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return str(data.get("message", {}).get("content", "")).strip()


def summarize_frame_with_model(
    image_path: Path,
    timestamp_label: str,
    model: str = DEFAULT_VISION_MODEL,
    emit: VideoLogger | None = None,
) -> dict[str, Any]:
    prompt = (
        "You are analyzing one frame from a bird video.\n"
        "Describe only what is visible in this frame.\n"
        "Do not invent titles, locations, or external context.\n"
        "Be careful and conservative.\n"
        "If you are unsure about a species, use unknown rather than guessing.\n"
        "Return plain text using exactly these lines:\n"
        "count_guess: <integer or unknown>\n"
        "species_guess: <short list of bracketed names like [Adélie Penguin], [Gentoo Penguin], or unknown>\n"
        "confidence: <low|medium|high>\n"
        "notes: <one short sentence describing what you actually see>\n"
        "If you cannot confidently count, use unknown.\n"
        "Do not output JSON."
    )
    raw_text = ollama_chat_with_image(model=model, prompt=prompt, image_path=image_path)
    observation = raw_text.strip()
    individual_count_guess = _extract_count_from_text(observation)
    species_guess_text = _extract_field_value(observation, "species_guess")
    confidence_value = _extract_confidence_from_text(observation)
    species_candidates = _extract_species_candidates(observation)

    if emit:
        emit(f"[local-video] frame note at {timestamp_label}: {observation}")

    return {
        "timestamp": timestamp_label,
        "frame": image_path.name,
        "visible_species_count": individual_count_guess,
        "species_guess": species_guess_text,
        "species_candidates": species_candidates,
        "confidence": confidence_value,
        "notes": observation,
    }


def build_video_state_report_bundle(
    url: str,
    task_id: str,
    frame_interval_seconds: int = 1,
    max_frames: int = 0,
    model: str = DEFAULT_VISION_MODEL,
    emit: VideoLogger | None = None,
) -> dict[str, str]:
    if emit:
        emit("[local-video] starting local video analysis")
    video_path = download_low_quality_youtube_video(url=url, task_id=task_id, emit=emit)
    frame_paths = sample_video_frames(
        video_path=video_path,
        task_id=task_id,
        frame_interval_seconds=frame_interval_seconds,
        max_frames=max_frames,
        emit=emit,
    )

    if not frame_paths:
        empty_report = "Local video analysis: no frames were extracted."
        return {"verbose": empty_report, "compact": empty_report}

    lines: list[str] = []
    concise_points: list[str] = []
    peak_individual_count: int | None = None
    peak_species_count: int | None = None
    for index, frame_path in enumerate(frame_paths):
        timestamp_seconds = index * frame_interval_seconds
        timestamp_label = f"{timestamp_seconds // 60:02d}:{timestamp_seconds % 60:02d}"
        if emit:
            emit(f"[local-video] analyzing frame {index + 1}/{len(frame_paths)} at {timestamp_label}")
        summary = summarize_frame_with_model(frame_path, timestamp_label=timestamp_label, model=model, emit=emit)
        individual_count_value = summary.get("visible_species_count")
        species_guess_value = str(summary.get("species_guess", "")).strip()
        species_candidates = list(summary.get("species_candidates", []))
        confidence_value = str(summary.get("confidence", "unknown")).strip().lower()
        if isinstance(individual_count_value, int):
            peak_individual_count = (
                individual_count_value
                if peak_individual_count is None
                else max(peak_individual_count, individual_count_value)
            )
        if species_candidates and confidence_value in {"medium", "high"}:
            species_count_value = len(species_candidates)
            peak_species_count = (
                species_count_value if peak_species_count is None else max(peak_species_count, species_count_value)
            )
            concise_points.append(f"{timestamp_label} -> {species_guess_value}")
        elif confidence_value == "high" and species_guess_value:
            concise_points.append(f"{timestamp_label} -> {species_guess_value}")
        lines.append(
            f"{summary['timestamp']} | frame={summary['frame']} | "
            f"individuals={individual_count_value} | species_guess={species_guess_value or 'unknown'} | "
            f"species_count={len(species_candidates) if species_candidates else 'unknown'} | "
            f"confidence={confidence_value} | notes={summary.get('notes')}"
        )

    verbose_report_lines = [
        "Local video analysis summary:",
        f"- source: {url}",
        f"- sampled at: {frame_interval_seconds}s intervals",
        f"- frames analyzed: {len(frame_paths)}",
        "- species count rule: only bracketed species candidates with medium/high confidence are counted",
        f"- peak distinct bird species in sampled frames: {peak_species_count if peak_species_count is not None else 'unknown'}",
        f"- peak bird individual count guess: {peak_individual_count if peak_individual_count is not None else 'unknown'}",
        f"- concise timeline: {', '.join(concise_points) if concise_points else 'none'}",
        "- per-frame timeline:",
        *[f"  {line}" for line in lines],
    ]
    compact_report_lines = [
        "Local video analysis summary:",
        f"- sampled at: {frame_interval_seconds}s intervals",
        f"- frames analyzed: {len(frame_paths)}",
        "- species count rule: only bracketed species candidates with medium/high confidence are counted",
        f"- peak distinct bird species in sampled frames: {peak_species_count if peak_species_count is not None else 'unknown'}",
        f"- peak bird individual count guess: {peak_individual_count if peak_individual_count is not None else 'unknown'}",
        f"- concise timeline: {', '.join(concise_points) if concise_points else 'none'}",
    ]
    if emit:
        emit("[local-video] analysis summary ready")
        emit(f"[local-video] peak sampled species count: {peak_species_count if peak_species_count is not None else 'unknown'}")
    return {
        "verbose": "\n".join(verbose_report_lines),
        "compact": "\n".join(compact_report_lines),
    }


def build_video_state_report(
    url: str,
    task_id: str,
    frame_interval_seconds: int = 1,
    max_frames: int = 0,
    model: str = DEFAULT_VISION_MODEL,
    emit: VideoLogger | None = None,
) -> str:
    return build_video_state_report_bundle(
        url=url,
        task_id=task_id,
        frame_interval_seconds=frame_interval_seconds,
        max_frames=max_frames,
        model=model,
        emit=emit,
    )["verbose"]


def cache_report_path(task_id: str) -> Path:
    return ensure_cache_dir(task_id) / "state_report.txt"


def load_cached_video_report(task_id: str) -> str | None:
    report_path = cache_report_path(task_id)
    if report_path.exists():
        return report_path.read_text(encoding="utf-8")
    return None


def save_cached_video_report(task_id: str, report: str) -> Path:
    report_path = cache_report_path(task_id)
    report_path.write_text(report, encoding="utf-8")
    return report_path
