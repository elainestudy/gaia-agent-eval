import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from agent import build_agent_prompt, run_agent
from tools.file_router import build_attachment_evidence, download_attachment
from tools.video_local import (
    build_video_state_report_bundle,
    extract_youtube_url,
    load_cached_video_report,
    save_cached_video_report,
)

API_BASE = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_OUTPUT_FILE = Path("my_submission.jsonl")
DEFAULT_ATTACHMENT_DIR = Path("task_files")

ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:answer|final answer|final|result|response)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)")
LIST_QUESTION_HINTS = (
    "comma separated list",
    "comma-separated list",
    "alphabetize",
    "alphabetized",
    "list of just",
    "place each item",
)


def fetch_json(url: str) -> Any:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def post_json(url: str, payload: dict[str, Any]) -> Any:
    response = requests.post(url, json=payload, timeout=120)
    if not response.ok:
        print(f"POST {url} failed with status {response.status_code}")
        print(response.text)
    response.raise_for_status()
    return response.json()


def get_random_question() -> dict[str, Any]:
    return fetch_json(f"{API_BASE}/random-question")


def get_all_questions() -> list[dict[str, Any]]:
    questions = fetch_json(f"{API_BASE}/questions")
    if not isinstance(questions, list):
        raise TypeError("Expected /questions to return a list of question objects.")
    return questions


def get_question_by_task_id(task_id: str) -> dict[str, Any]:
    questions = get_all_questions()
    normalized_task_id = str(task_id)
    for question in questions:
        if str(question.get("task_id", "")) == normalized_task_id:
            return question
    raise ValueError(f"Task id not found in /questions: {task_id}")


def question_file_name(question: dict[str, Any]) -> str | None:
    file_name = question.get("file_name")
    return str(file_name) if file_name else None


def question_has_attachment(question: dict[str, Any]) -> bool:
    attachment_keys = (
        "has_file",
        "file",
        "file_url",
        "file_id",
        "file_name",
        "attachment",
        "attachments",
        "has_attachment",
    )
    for key in attachment_keys:
        value = question.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (str, list, dict)) and value:
            return True
    return False


def download_task_file(
    task_id: str, output_dir: Path = DEFAULT_ATTACHMENT_DIR, file_name: str | None = None
) -> Path | None:
    file_path, _ = download_attachment(task_id=task_id, api_base=API_BASE, emit=print, file_name=file_name)
    return file_path


def enrich_question_with_attachment(question: dict[str, Any]) -> str:
    base_question = str(question.get("question", ""))
    task_id = str(question.get("task_id", ""))

    if not task_id or not question_has_attachment(question):
        return base_question

    file_path = download_task_file(task_id)
    if file_path is None:
        return base_question

    try:
        if file_path.suffix == ".json":
            attachment_text = file_path.read_text(encoding="utf-8")
        else:
            attachment_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"{base_question}\n\nAttachment downloaded to: {file_path}"

    return f"{base_question}\n\nAttachment content:\n{attachment_text}"


def enrich_question_with_local_video_state(question: dict[str, Any]) -> str:
    base_question = str(question.get("question", ""))
    task_id = str(question.get("task_id", ""))
    youtube_url = extract_youtube_url(base_question)

    if not youtube_url or not task_id:
        return ""

    cached_report = load_cached_video_report(task_id)
    if cached_report:
        print("[local-video] using cached analysis report")
        return compact_video_report(cached_report)

    try:
        report_bundle = build_video_state_report_bundle(
            url=youtube_url,
            task_id=task_id,
            frame_interval_seconds=3,
            max_frames=0,
            emit=print,
        )
        save_cached_video_report(task_id, report_bundle["verbose"])
        return report_bundle["compact"]
    except Exception as exc:
        print(f"[local-video] analysis failed: {exc}")
        return f"Local video analysis unavailable: {exc}"


def compact_video_report(report: str) -> str:
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    compact_lines: list[str] = []

    for line in lines:
        if line.startswith("Local video analysis summary:"):
            compact_lines.append(line)
        elif line.startswith("- sampled at:"):
            compact_lines.append(line)
        elif line.startswith("- frames analyzed:"):
            compact_lines.append(line)
        elif line.startswith("- peak distinct bird species in sampled frames:"):
            compact_lines.append(line)
        elif line.startswith("- peak bird individual count guess:"):
            compact_lines.append(line)
        elif line.startswith("- concise timeline:"):
            compact_lines.append(line)

    if not compact_lines:
        return report

    return "\n".join(compact_lines)


def solve_question(question: dict[str, Any], verbose: bool = False, diagnostics: dict[str, Any] | None = None) -> str:
    base_question = str(question.get("question", ""))
    task_id = str(question.get("task_id", ""))

    evidence_blocks: list[str] = [base_question]

    if question_has_attachment(question):
        attachment_evidence = build_attachment_evidence(
            api_base=API_BASE,
            task_id=task_id,
            question_text=base_question,
            emit=print if verbose else None,
            file_name=question_file_name(question),
        )
        evidence_blocks.append(attachment_evidence)

    video_evidence = enrich_question_with_local_video_state({"task_id": task_id, "question": base_question})
    if video_evidence:
        evidence_blocks.append(video_evidence)

    task_notes = (
        "Use the observation-classification-knowledge lookup workflow.\n"
        "Textual attachments should be parsed directly.\n"
        "Visual attachments should be summarized by the local vision model.\n"
        "Search only when the category is ambiguous or evidence conflicts."
    )
    prompt = build_agent_prompt(
        question=base_question,
        evidence_blocks=evidence_blocks[1:],
        task_notes=task_notes,
    )
    return run_agent(prompt, verbose=verbose, diagnostics=diagnostics)


def write_jsonl(results: list[dict[str, Any]], output_file: Path = DEFAULT_OUTPUT_FILE) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


NUMERIC_QUESTION_HINTS = (
    "how many",
    "how much",
    "what is the number",
    "what number",
    "what is the total",
    "total number",
    "total sales",
    "highest number",
    "lowest number",
    "count",
    "many",
    "sum",
    "difference",
    "ratio",
    "percentage",
)


def clean_model_answer(answer: Any, question_text: str | None = None) -> str:
    text = str(answer).strip()
    if not text:
        return ""

    question_lower = question_text.lower() if question_text else ""
    numeric_question = any(hint in question_lower for hint in NUMERIC_QUESTION_HINTS)
    list_question = any(hint in question_lower for hint in LIST_QUESTION_HINTS)

    fenced_blocks = re.findall(r"```(?:[\w+-]+)?\s*([\s\S]*?)```", text)
    if fenced_blocks:
        text = fenced_blocks[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        numeric_candidates = NUMBER_PATTERN.findall(text)
        if numeric_question and numeric_candidates:
            return numeric_candidates[-1]
        if list_question:
            for line in reversed(lines):
                candidate = ANSWER_PREFIX_PATTERN.sub("", line).strip()
                candidate = candidate.strip("`'\"“”*")
                if "," in candidate or ";" in candidate:
                    if ":" in candidate:
                        tail = candidate.split(":", maxsplit=1)[-1].strip()
                        if "," in tail or ";" in tail:
                            candidate = tail
                    return candidate
        for line in reversed(lines):
            candidate = ANSWER_PREFIX_PATTERN.sub("", line).strip()
            candidate = candidate.strip("`'\"“”*")
            if candidate:
                return candidate

    numeric_candidates = NUMBER_PATTERN.findall(text)
    if numeric_question and numeric_candidates:
        return numeric_candidates[-1]

    candidate = ANSWER_PREFIX_PATTERN.sub("", text).strip()
    candidate = candidate.strip("`'\"“”*")
    return candidate


def load_existing_task_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    existing_task_ids: set[str] = set()
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id is not None:
                existing_task_ids.add(str(task_id))
    return existing_task_ids


def test_single_random_question(verbose: bool = True) -> None:
    print("🎲 Fetching one random question...")
    question = get_random_question()

    task_id = question.get("task_id")
    print(f"\n[task_id: {task_id}]")
    print(f"question: {question.get('question')}")
    print(f"attachment_detected: {question_has_attachment(question)}")

    if task_id and question_has_attachment(question):
        file_path = download_task_file(str(task_id), file_name=question_file_name(question))
        print(f"attachment_downloaded_to: {file_path}")

    print("\n🤖 Running agent...")
    try:
        answer = solve_question(question, verbose=verbose)
        print(f"\nanswer: {answer}")
        print(f"clean_answer: {clean_model_answer(answer, str(question.get('question', '')))}")
    except Exception as exc:
        print(f"\n[test-one failed] {exc}")
        raise


def append_jsonl_record(output_file: Path, record: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_answer_jsonl(output_file: Path = DEFAULT_OUTPUT_FILE, verbose: bool = False) -> list[dict[str, Any]]:
    print("🚀 Fetching all questions...")
    questions = get_all_questions()
    print(f"Found {len(questions)} questions.")

    completed_task_ids = load_existing_task_ids(output_file)
    if completed_task_ids:
        print(f"Resuming from existing file. Found {len(completed_task_ids)} completed task_id values.")

    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        task_id = str(question.get("task_id", ""))

        if task_id in completed_task_ids:
            print(f"\n[{index}/{len(questions)}] Skipping already completed task {task_id}.")
            continue

        print(f"\n[{index}/{len(questions)}] Solving task {task_id}...")

        try:
            diagnostics: dict[str, Any] = {}
            answer = solve_question(question, verbose=verbose, diagnostics=diagnostics)
            clean_answer = clean_model_answer(answer, str(question.get("question", "")))
            record = {
                "task_id": task_id,
                "model_answer": clean_answer,
                "used_fallback": diagnostics.get("used_fallback", False),
            }
            append_jsonl_record(output_file, record)
            results.append(record)
            completed_task_ids.add(task_id)
            print(f"Saved answer for task {task_id}.")
        except Exception as exc:
            print(f"Failed task {task_id}: {exc}")
            print(f"Question text: {question.get('question')}")
            # Do not mark this task_id as completed and do not write a record: a blank
            # answer would otherwise get silently resubmitted to the real, rate-limited
            # scoring endpoint by a later `submit-all` run. Leaving it out of the file
            # means the next `build-jsonl` run retries it instead.

    print(f"\nWrote clean JSONL to {output_file.resolve()}")
    return results


def load_agent_code(agent_path: Path) -> str:
    return agent_path.read_text(encoding="utf-8")


def submit_answers(username: str, agent_code: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": [
            {
                "task_id": item["task_id"],
                "submitted_answer": item["model_answer"],
            }
            for item in answers
        ],
    }
    return post_json(f"{API_BASE}/submit", payload)


def submit_one_question(username: str, agent_path: Path, task_id: str | None = None, verbose: bool = True) -> None:
    if task_id:
        question = get_question_by_task_id(task_id)
        print(f"Testing submit with fixed task_id={task_id}")
    else:
        question = get_random_question()
        task_id = str(question.get("task_id", ""))
        print(f"Testing submit with random task_id={task_id}")
    base_question = str(question.get("question", ""))

    print(f"Question text: {question.get('question')}")
    print(f"Attachment detected: {question_has_attachment(question)}")

    try:
        answer = solve_question(question, verbose=verbose)
    except Exception as exc:
        print(f"[submit-one failed during solve] {exc}")
        raise

    agent_code = load_agent_code(agent_path)
    response = submit_answers(
        username=username,
        agent_code=agent_code,
        answers=[{"task_id": task_id, "model_answer": clean_model_answer(answer, base_question)}],
    )
    print(f"clean_answer: {clean_model_answer(answer, base_question)}")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def submit_all_answers(username: str, agent_path: Path, output_file: Path = DEFAULT_OUTPUT_FILE) -> None:
    if not output_file.exists():
        raise FileNotFoundError(f"{output_file} does not exist. Run --mode build-jsonl first.")

    records: list[dict[str, Any]] = []
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"{output_file} has no records to submit.")

    blank_task_ids = [r["task_id"] for r in records if not str(r.get("model_answer", "")).strip()]
    if blank_task_ids:
        print(
            f"Skipping {len(blank_task_ids)} record(s) with a blank model_answer "
            f"(re-run build-jsonl to fill these in first): {blank_task_ids}"
        )
    records = [r for r in records if str(r.get("model_answer", "")).strip()]
    if not records:
        raise ValueError(f"{output_file} has no non-blank records to submit.")

    print(f"Submitting {len(records)} answers from {output_file}...")
    agent_code = load_agent_code(agent_path)
    response = submit_answers(
        username=username,
        agent_code=agent_code,
        answers=[{"task_id": r["task_id"], "model_answer": r["model_answer"]} for r in records],
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA evaluation pipeline")
    parser.add_argument(
        "--mode",
        choices=("test-one", "build-jsonl", "submit-one", "submit-all"),
        default="test-one",
        help=(
            "test-one: run one random question, build-jsonl: solve all questions and write JSONL, "
            "submit-one: submit one solved question to /submit, "
            "submit-all: submit every answer already in the JSONL output file to /submit"
        ),
    )
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--username", default="")
    parser.add_argument("--agent-path", default="agent.py")
    parser.add_argument("--task-id", default="", help="Use a fixed task id for submit-one mode.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose agent tracing.")
    args = parser.parse_args()

    output_file = Path(args.output_file)
    agent_path = Path(args.agent_path)

    if args.mode == "test-one":
        test_single_random_question(verbose=not args.quiet)
        return

    if args.mode == "build-jsonl":
        build_answer_jsonl(output_file, verbose=not args.quiet)
        return

    if not args.username:
        raise ValueError(f"--username is required when --mode {args.mode}")

    if args.mode == "submit-all":
        submit_all_answers(args.username, agent_path, output_file)
        return

    submit_one_question(args.username, agent_path, task_id=args.task_id or None, verbose=not args.quiet)


if __name__ == "__main__":
    main()
