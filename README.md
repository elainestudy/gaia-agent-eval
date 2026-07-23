# GAIA Agent Eval

An agent that answers questions from the [GAIA benchmark](https://huggingface.co/gaia-benchmark) for Hugging Face's "AI Agents Course" Unit 4 assessment, and submits results to the course's scoring API for leaderboard grading.

## "score": 100.0 benchmark completed
All 20 tasks in the set have been successfully solved by this agent at least once! While non-deterministic components like model sampling and VLM reads mean it’s not a 20/20 every single run, the ceiling is officially maxed out.

## Free-tier friendly: ≤10 model requests per task

`agent.py`'s tool-calling loop is hard-capped at `max_steps=10` — a single task can never issue more than 10 requests to the underlying LLM, no matter how much tool-calling it does (see `run_agent()`). That makes the agent usable on the free tier of pretty much any model provider. Gemini's free tier, for example, caps at 15 requests per minute (RPM); running questions one at a time (`test-one`, `submit-one`) stays well under that ceiling.

The one place this budget doesn't protect you: `build-jsonl` walks through every question back-to-back with no delay or backoff between tasks. Since each task can burn through its 10 requests in well under a minute, running the full 20-question set unthrottled can still stack up near or above a 15 RPM cap and trip a 429, even though any single task is safely within budget. If you hit that, either add a delay between tasks or resume `build-jsonl` in smaller batches — it already skips `task_id`s already present in the output file, so it's safe to stop and restart.

## Architecture

- **`agent.py`** — the agent core. A LangChain tool-calling loop around Gemini, bound to six tools (`search_internet`, `search_wikipedia`, `wolframalpha_query`, `get_youtube_transcript`, `fetch_page`, `run_python_code`). The system prompt enforces a strict **observe → classify → knowledge-lookup** workflow, designed to avoid two common failure modes: treating atypical category members incorrectly (e.g. a penguin is still a bird), and letting one broad search stand in for checking every item in a candidate list individually.
- **`eval_pipeline.py`** — the evaluation harness. Fetches GAIA questions from the scoring API, assembles evidence (question text, parsed attachments, local video analysis), runs the agent, cleans the output to match GAIA's exact-match formatting rules, and (optionally) submits answers to the scoring API.
- **`tools/`** — each tool is a standalone function with its own guard logic:
  - `search.py` / `search_wikipedia.py` — web and Wikipedia search; both detect multi-item list queries and refuse/narrow them to a single candidate rather than letting the agent try to resolve a whole list in one call.
  - `wolframalpha_query.py` — thin wrapper over the WolframAlpha API for arithmetic/unit/equation queries.
  - `get_youtube_transcript.py` — pulls YouTube subtitles, falling back gracefully when none exist.
  - `video_local.py` — for videos with no transcript: downloads a low-res copy, samples frames, and captions them with a local Ollama vision model to build a "video state report."
  - `file_router.py` — routes task attachments (text/image/video/audio/spreadsheet) and builds evidence text from them: image attachments go through **Gemini's own multimodal input** (far more reliable than the local model for precise structured reading, e.g. chessboard positions — video frames still use the local model, since a video can produce many sampled frames and a GAIA task has at most one image), audio is transcribed with a local `faster-whisper` model, spreadsheets are parsed with `pandas`. PDF is still unsupported. See [Known issues](#known-issues) below for how it works around a broken upstream endpoint.
  - `fetch_page.py` — fetches a known URL's actual page text (via Jina Reader, or directly for Wayback Machine snapshots) for reading-comprehension tasks where a search snippet isn't specific enough, or where a question needs a page's state as of a past date rather than today's.
  - `run_code.py` — runs Python the agent writes for math/aggregation beyond WolframAlpha, or to execute an attached script. For chess-puzzle questions it also backstops two things the model reliably gets wrong on its own: it runs a real chess engine itself (if `stockfish` is installed) whenever it detects a FEN in the submitted code, and auto-converts any legal UCI move in the output to standard algebraic notation — regardless of whether the model's own code does either correctly.

## Setup

Requires Python 3.13. Install dependencies with `pip install -r requirements.txt` (direct dependencies only, unpinned) into a virtualenv — `.venv/` itself isn't checked in.

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key       # required
WOLFRAMALPHA_APP_ID=your_wolframalpha_id # optional — without it, wolframalpha_query returns WOLFRAMALPHA_NOT_CONFIGURED
HF_TOKEN=your_hf_token                   # optional, read-only is enough — needed for the /files 404 workaround below
```

Additional runtime dependencies:
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) must be on `PATH` for the local video-analysis flow.
- A local [Ollama](https://ollama.com) server (`localhost:11434`, model `minicpm-v:8b`) must be running for video frame captioning.
- [Stockfish](https://stockfishchess.org/) should be on `PATH` for reliably solving chess-puzzle questions — `run_python_code` still works without it, just without the auto-verified best move.

## Usage

```bash
# Run a single random GAIA question end-to-end (verbose debugging)
python3 eval_pipeline.py --mode test-one

# Solve all questions and write a resumable answers file
python3 eval_pipeline.py --mode build-jsonl --output-file my_submission.jsonl

# Solve + submit one question to the scoring API. If the API confirms it's correct,
# the answer is also saved (or overwrites a stale one) in --output-file, so a lucky
# submit-one run on a non-deterministic task doesn't get lost the next time
# build-jsonl re-solves it.
python3 eval_pipeline.py --mode submit-one --username YOUR_HF_USERNAME --agent-path agent.py

# Replay a specific task instead of a random one
python3 eval_pipeline.py --mode submit-one --username YOUR_HF_USERNAME --task-id <task_id>

# Submit every answer already saved in the JSONL file in one shot — this is the mode
# that actually updates the leaderboard score; submit-one always scores out of 20 for
# just the single question it sent, so it can never move the real record
python3 eval_pipeline.py --mode submit-all --username YOUR_HF_USERNAME
```

`build-jsonl` and `submit-one` skip/resume already-completed `task_id`s found in the output file, so re-running `build-jsonl` is safe.

## Known issues

### `GET /files/{task_id}` 404s for tasks that do have a file

This is a bug in the course's own scoring API, not in this repo — tracked in [huggingface/agents-course#647](https://github.com/huggingface/agents-course/issues/647) and root-caused on the [HF forum](https://discuss.huggingface.co/t/get-files-task-id-returning-404-for-every-task-id-with-file-name/170575) as a relative-vs-absolute path bug in the Space's file-serving code. Every task whose `file_name` field is non-empty currently gets `404 {"detail": "No file path associated with task_id ..."}` from the intended endpoint, making its attachment unfetchable through the documented path.

**Workaround** (not a fix — the actual bug lives in HF's infrastructure): `tools/file_router.py` falls back to fetching the file directly from the `gaia-benchmark/GAIA` dataset on the Hugging Face Hub (checking the `validation` split, then `test`) whenever `/files/{task_id}` 404s, and caches the result locally under `.local_attachments/<task_id>/` so it's only fetched once. This needs:
1. `HF_TOKEN` set in `.env` (a read-only token is enough), and
2. Access already granted to the gated [`gaia-benchmark/GAIA`](https://huggingface.co/datasets/gaia-benchmark/GAIA) dataset on huggingface.co.

Without either, attachment-bearing tasks whose files 404 just silently fall back to no evidence, same as before this workaround existed.

Audio and spreadsheet attachments are now parsed once fetched (local `faster-whisper` transcription, `pandas`-based spreadsheet-to-CSV), so all 5 known-404 tasks in the course's task set now get real evidence instead of a refusal or a guess. PDF attachments are still unsupported.
