# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agent that answers **GAIA benchmark** questions for the Hugging Face "AI Agents Course" Unit 4 assessment, and submits results to the course's scoring API (`agents-course-unit4-scoring.hf.space`) for leaderboard grading.

## Environment & running

Dependencies are listed in `requirements.txt` (direct dependencies only, unpinned — install with `pip install -r requirements.txt`). `.venv` (Python 3.13) is gitignored, not checked in; it's the local install target, not the source of truth. `chess` (python-chess) isn't imported by any repo file directly -- it's installed so `run_python_code` can import it for board/legal-move reasoning on chess-puzzle tasks. Use `.venv/bin/python3` / `.venv/bin/pip` directly, or activate the venv first.

Requires a `.env` file with:
- `GOOGLE_API_KEY` — required, Gemini access for `agent.py`
- `WOLFRAMALPHA_APP_ID` — optional; without it `wolframalpha_query` just returns `WOLFRAMALPHA_NOT_CONFIGURED`
- `HF_TOKEN` — optional but needed for the `/files/{task_id}` 404 workaround below to actually fetch anything; a read-only token is enough, and you must have accepted access to the gated `gaia-benchmark/GAIA` dataset on huggingface.co first

`yt-dlp` must be on `PATH` for the local video-analysis flow, and a local **Ollama** server (`localhost:11434`, model `minicpm-v:8b`) must be running for image/video frame captioning.

No test suite or lint config exists in this repo.

### Common commands

```bash
# Run a single random GAIA question end-to-end (debugging, verbose by default)
python3 eval_pipeline.py --mode test-one

# Solve all questions and write a resumable answers file
python3 eval_pipeline.py --mode build-jsonl --output-file my_submission.jsonl

# Solve + submit one question to the scoring API
python3 eval_pipeline.py --mode submit-one --username YOUR_HF_USERNAME --agent-path agent.py

# Replay a specific task instead of a random one (works with test-one via code, or submit-one via flag)
python3 eval_pipeline.py --mode submit-one --username YOUR_HF_USERNAME --task-id <task_id>

# Submit every answer already saved in the JSONL output file in one shot (this is
# what actually updates the leaderboard score — submit-one only ever scores out of
# 20 for a single question, so it can never move the real record)
python3 eval_pipeline.py --mode submit-all --username YOUR_HF_USERNAME

# Manually pull a low-res video into the local cache for the video-analysis flow
yt-dlp -f "worstvideo" -o ".cache/video/%(id)s/%(id)s.%(ext)s" "<youtube_url>"
```

`build-jsonl` and `submit-one` skip/resume already-completed `task_id`s found in the output file, so re-running `build-jsonl` is safe.

## Architecture

**`agent.py`** — the agent core. LangChain tool-calling loop (`run_agent`) around Gemini (`gemini-3.1-flash-lite`). Bound tools: `search_internet`, `search_wikipedia`, `wolframalpha_query`, `get_youtube_transcript`, `fetch_page`, `run_python_code`. The prompts (`SYSTEM_PROMPT`, `OBSERVATION_PROMPT_TEMPLATE`, `CLASSIFICATION_PROMPT_TEMPLATE`, `TOOL_SELECTION_PROMPT_TEMPLATE`, etc., composed via `build_agent_prompt`) enforce a strict **observe → classify → knowledge-lookup** workflow. This is deliberate and load-bearing: it exists to fix specific GAIA failure modes — treating atypical category members correctly (e.g. penguins are birds), never letting one broad search substitute for checking every item in a candidate list one at a time, and choosing the right tool for the question type (Wikipedia = stable definitions/category membership, WolframAlpha = arithmetic/units/equations, web search = fallback/recency). When touching these prompts, preserve that separation of concerns rather than collapsing it.

**`eval_pipeline.py`** — the harness around the agent, calling the GAIA scoring API:
- Fetches questions (`/random-question`, `/questions`, or by `task_id`).
- `solve_question()` assembles "evidence blocks": the raw question, any attachment content (via `tools/file_router.py`), and any local video analysis (via `tools/video_local.py`) — then hands it all to `build_agent_prompt` + `run_agent`.
- `clean_model_answer()` post-processes raw model output before submission: strips code fences/answer prefixes, pulls the trailing number for numeric-hint questions, extracts comma/semicolon-separated lists for list-hint questions. This exists because GAIA scoring requires exact-match formatting — don't bypass it when changing answer generation.
- `submit_answers()` POSTs `{username, agent_code, answers: [{task_id, submitted_answer}]}` — note the payload key is `submitted_answer` but the local JSONL/record key is `model_answer`.

**`tools/`** — each tool is a standalone `@tool`-decorated function with its own guard logic to keep the agent honest:
- `search.py` / `search_wikipedia.py` both detect "multi-item list" queries (comma-heavy) and refuse/narrow them to a single candidate, because the agent tends to try to resolve an entire candidate list with one query instead of checking items individually. `search_wikipedia.py` additionally refuses yes/no classification queries, redirecting them to `search_internet`.
- `wolframalpha_query.py` — thin wrapper over the WolframAlpha `v1/result` API.
- `get_youtube_transcript.py` — extracts subtitles via `youtube_transcript_api`; returns `NO_TRANSCRIPT_AVAILABLE` on any failure rather than raising, so the agent can fall back to other tools.
- `video_local.py` — for YouTube videos with no usable transcript: downloads a low-res copy with `yt-dlp`, samples frames, and captions them via a **local Ollama vision model** (`ollama_chat_with_image`, default `minicpm-v:8b`), building a "video state report" (e.g. tracking entity counts/species over time). Reports are cached under `.cache/video/<task_id>/` and reused on rerun.
- `file_router.py` — routes task attachments by detected type (text/image/video/audio/spreadsheet/pdf via extension or byte sniffing) and builds evidence text accordingly; images/video frames go through the same local vision model as `video_local.py`, audio is transcribed with a local `faster-whisper` model, spreadsheets are parsed with `pandas`. Caches under `.cache/attachments/<task_id>/` for the official API path, `.local_attachments/<task_id>/` for the Hub-fallback path below.
- `fetch_page.py` — fetches a known URL's actual page text via Jina Reader (`r.jina.ai`), for reading-comprehension tasks where a search snippet isn't enough. Blocks requests to private/local network addresses, and gives Wikipedia pages a larger truncation budget plus a `X-Target-Selector` header so nav/language-switcher chrome doesn't crowd out the real article body. `web.archive.org` URLs are fetched directly instead (Jina 403s on Wayback) and parsed with `lxml` — the target page's original host still gets the Wikipedia/LibreTexts content-selector treatment (via `lxml`'s `get_element_by_id`, since there's no Jina header to do it server-side) and the same Wikipedia truncation budget.
- `run_code.py` — runs an arbitrary Python snippet via `subprocess` (`sys.executable`, 10s timeout) for aggregation/math the agent shouldn't do in its head. Its "guard logic" is a static denylist rejecting code that imports `os`/`subprocess`/`socket`/etc. or calls `open()`/`eval()`/`exec()` — it keeps the tool scoped to pure computation, but this is not a real sandbox (no resource/network isolation); treat it as trusted-input convenience, not a security boundary.

## Known upstream issue: `/files/{task_id}` 404s for tasks that do have a file

This is a real, still-open bug in the course's scoring API, not something in this repo:  `GET /files/{task_id}` returns 404 (`"No file path associated with task_id ..."`) for every task whose `file_name` field is non-empty, i.e. attachments are effectively unfetchable through the intended endpoint. Tracked at [huggingface/agents-course#647](https://github.com/huggingface/agents-course/issues/647), root-caused on the [HF forum](https://discuss.huggingface.co/t/get-files-task-id-returning-404-for-every-task-id-with-file-name/170575) as a relative-vs-absolute path bug in the Space's own file-serving code.

**Workaround, not a fix** (the actual bug lives in HF's Space and isn't something this repo can patch): `tools/file_router.py`'s `download_attachment()` falls back to fetching the file directly from the `gaia-benchmark/GAIA` dataset on the Hub (trying the `validation` split first, then `test`) whenever `/files/{task_id}` 404s, and caches the result under `.local_attachments/<task_id>/` so later runs don't hit the Hub again. This needs `HF_TOKEN` set and access to the gated dataset already granted; without either, attachment-bearing tasks just silently fall back to no evidence, same as before this workaround existed.

Audio (`.mp3`/`.wav`/`.m4a`/`.ogg`/`.flac`) is transcribed with a local `faster-whisper` model (`WHISPER_MODEL_SIZE = "base"`, CPU, cached as a module-level singleton so it's only loaded once per process); spreadsheets (`.xlsx`/`.xls`) are read with `pandas.read_excel` and every sheet is serialized to CSV text as evidence. PDF attachments are still unsupported.

## Notes carried over from local dev workflow

- `search_wikipedia` is for definitions/concept disambiguation; `wolframalpha_query` needs `WOLFRAMALPHA_APP_ID` to do anything.
- If a task has an attachment, the pipeline tries `/files/{task_id}` first, then the Hub fallback above, and appends parsed content to the prompt automatically — no manual step needed.
- Numeric answers should stay short and bare; the cleaning step in `eval_pipeline.py` depends on this.
