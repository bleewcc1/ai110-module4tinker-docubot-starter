# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DocuBot is a CodePath AI110 "tinker" starter: a small documentation assistant over the text files in `docs/`. It has three modes: naive LLM (Gemini, no retrieval), retrieval-only, and RAG (retrieval + Gemini). There is no linter or build step; the only tests are the retrieval unit tests in `tests/`.

## Commands

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set GEMINI_API_KEY
python main.py                # interactive CLI: pick mode 1/2/3 or q
python evaluation.py          # retrieval hit rate over SAMPLE_QUERIES (no LLM needed)
python -m unittest discover tests -v   # retrieval unit tests (stdlib, no API key)
```

Retrieval-only mode and `evaluation.py` work without an API key; modes 1 and 3 need `GEMINI_API_KEY`.

## Architecture

- [main.py](main.py): CLI loop. Calls `load_dotenv()` first, builds a `GeminiClient` (falls back to no-LLM if the key is missing), then dispatches to the mode runners.
- [docubot.py](docubot.py): `DocuBot` loads `docs/*.md|*.txt` as `(filename, text)` tuples and owns retrieval. `build_index`, `score_document`, and `retrieve` are **TODO stubs** (Phase 1) that currently return empty results. So retrieval-only mode answers "I do not know" and RAG short-circuits before calling Gemini until they are implemented. Retrieval returns whole documents, not chunks.
- [llm_client.py](llm_client.py): `GeminiClient` wraps `google-genai`; model name is the single constant `GEMINI_MODEL_NAME`. `naive_answer_over_full_docs` deliberately ignores the docs text (Phase 0 baseline, to show hallucination). `answer_from_snippets` builds the strict "answer only from snippets, else reply exactly 'I do not know based on the docs I have.'" prompt. API failures are caught and returned as strings, not raised, so they show up in the printed "answer".
- [evaluation.py](evaluation.py): scores `bot.retrieve` against `EXPECTED_SOURCES`, a substring-to-filename map. A query with no matching substring counts as a miss, so add entries there when adding queries to [dataset.py](dataset.py).
- [dataset.py](dataset.py): `SAMPLE_QUERIES`, plus `FALLBACK_DOCS` (not wired into `DocuBot`).

## Gotchas

- `load_dotenv()` does not override variables already in the environment, so a stale `GEMINI_API_KEY` exported in the shell or IDE beats `.env`. Use `unset GEMINI_API_KEY` if you get "API key not valid" despite a good `.env`.
- The project lives on an exFAT drive, which creates macOS `._*` AppleDouble files. Inside `.venv/.../site-packages` these can break `pip`/`site` with a `UnicodeDecodeError`. Delete them with `find .venv -name '._*' -delete`.
- `requirements.txt` pins `certifi>=2026.10.19` on purpose (past the revoked emSign intermediates); don't loosen it.
