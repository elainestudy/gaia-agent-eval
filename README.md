# GAIA Agent Eval

An agent that answers questions from the [GAIA benchmark](https://huggingface.co/gaia-benchmark) for Hugging Face's "AI Agents Course" Unit 4 assessment, and submits results to the course's scoring API for leaderboard grading.

## Architecture

- **`agent.py`** — the agent core. A LangChain tool-calling loop around Gemini, bound to four tools (`search_internet`, `search_wikipedia`, `wolframalpha_query`, `get_youtube_transcript`). The system prompt enforces a strict **observe → classify → knowledge-lookup** workflow, designed to avoid two common failure modes: treating atypical category members incorrectly (e.g. a penguin is still a bird), and letting one broad search stand in for checking every item in a candidate list individually.
- **`eval_pipeline.py`** — the evaluation harness. Fetches GAIA questions from the scoring API, assembles evidence (question text, parsed attachments, local video analysis), runs the agent, cleans the output to match GAIA's exact-match formatting rules, and (optionally) submits answers to the scoring API.
- **`tools/`** — each tool is a standalone function with its own guard logic:
  - `search.py` / `search_wikipedia.py` — web and Wikipedia search; both detect multi-item list queries and refuse/narrow them to a single candidate rather than letting the agent try to resolve a whole list in one call.
  - `wolframalpha_query.py` — thin wrapper over the WolframAlpha API for arithmetic/unit/equation queries.
  - `get_youtube_transcript.py` — pulls YouTube subtitles, falling back gracefully when none exist.
  - `video_local.py` — for videos with no transcript: downloads a low-res copy, samples frames, and captions them with a local Ollama vision model to build a "video state report."
  - `file_router.py` — routes task attachments (text/image/video/pdf) and builds evidence text from them, reusing the same local vision model for images.

## Setup

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key       # required
WOLFRAMALPHA_APP_ID=your_wolframalpha_id # optional — without it, wolframalpha_query returns WOLFRAMALPHA_NOT_CONFIGURED
```

Additional runtime dependencies:
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) must be on `PATH` for the local video-analysis flow.
- A local [Ollama](https://ollama.com) server (`localhost:11434`, model `minicpm-v:8b`) must be running for image/video frame captioning.

## Usage

```bash
# Run a single random GAIA question end-to-end (verbose debugging)
python3 eval_pipeline.py --mode test-one

# Solve all questions and write a resumable answers file
python3 eval_pipeline.py --mode build-jsonl --output-file my_submission.jsonl

# Solve + submit one question to the scoring API
python3 eval_pipeline.py --mode submit-one --username YOUR_HF_USERNAME --agent-path agent.py

# Replay a specific task instead of a random one
python3 eval_pipeline.py --mode submit-one --username YOUR_HF_USERNAME --task-id <task_id>
```

`build-jsonl` and `submit-one` skip/resume already-completed `task_id`s found in the output file, so re-running `build-jsonl` is safe.
