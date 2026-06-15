# CLAUDE.md — PodcastCard

Context for AI assistants working on this repository.

## What this project is

**PodcastCard** is a tool for studying Mandarin Chinese from podcasts. You give it a
URL to a podcast/video episode; it downloads the audio, transcribes it, extracts the
Chinese words and phrases, and organizes them by HSK level (1–6) while preserving the
sentence context each word appeared in. The goal: a learner can review high-level
vocabulary alongside the sentence it was used in.

There are two ways to use it:
- **CLI** (`python -m src run <url>`) — prints results to the terminal and exports CSV.
- **Web UI** (`uvicorn src.app:app`) — single-page app with live progress, HSK filtering,
  expandable context sentences, and episode history.

## Tech stack

- **Language:** Python 3.10+ (uses `X | Y` union syntax and `from __future__ import annotations`)
- **Transcription:** `faster-whisper` (local Whisper, no API key, language pinned to `zh`)
- **Audio download:** `yt-dlp` (Python API, post-processes to mp3 via FFmpeg)
- **Segmentation:** `jieba` (Chinese word tokenizer)
- **Pinyin:** `pypinyin`
- **CLI:** `typer` + `rich`
- **Web:** `fastapi` + `uvicorn`, SSE for progress streaming
- **Storage:** `sqlite3` (stdlib, no ORM) — DB file `./podcastcard.db`
- **HSK data:** bundled `data/hsk_words.json` (~814 words, levels 1–6)

External system dependency: **FFmpeg** must be installed for audio decoding.

## Architecture / data flow

```
URL → audio.download_audio() → transcribe.transcribe() → extract.extract_words() → HSK lookup → CLI display / Web UI
```

## File map

| File | Responsibility |
|------|----------------|
| `src/audio.py` | `download_audio(url, output_dir) -> str` (path to mp3) via yt-dlp |
| `src/transcribe.py` | `transcribe(audio_path, model_size="base") -> list[Segment]` via faster-whisper |
| `src/extract.py` | `extract_words(segments) -> list[WordOccurrence]`; jieba tokenize + dedup + filter; defines `Segment` and `WordOccurrence` dataclasses |
| `src/hsk.py` | `get_hsk_level(word) -> int` (0 = unknown), `get_pinyin(word) -> str`, `HSK_WORDS` dict |
| `src/cli.py` | Typer app; `run` command; rich display + `words.csv` export |
| `src/app.py` | FastAPI server; REST + SSE routes; SQLite persistence |
| `src/__main__.py` | Entry point so `python -m src` runs the CLI |
| `static/index.html` | Single-page UI (vanilla HTML/CSS/JS, no build step, no frameworks) |
| `data/hsk_words.json` | `{word: hsk_level}` mapping |

## Key data shapes

```python
@dataclass
class Segment:
    start: float
    end: float
    text: str

@dataclass
class WordOccurrence:
    word: str
    pinyin: str
    hsk_level: int          # 0 = not in HSK list
    contexts: list[str]     # full sentences the word appeared in
```

Word extraction filters out: punctuation-only tokens, numbers, and common single-char
function words (的了是在我你他她它们这那和也都有不就把被从到与及). Results are sorted by
HSK level ascending, with unknown (level 0) words last.

## Web API routes (`src/app.py`)

- `GET /` — serves `static/index.html`
- `POST /analyze` — body `{url, model, hsk_levels}`; runs pipeline synchronously, returns JSON
- `GET /analyze/stream?url=&model=&hsk_levels=` — SSE; emits stages `downloading` →
  `transcribing` → `extracting` → `done` (full result on `done`), or `error`
- `GET /episodes` — list past episodes
- `GET /episodes/{id}/words?hsk_levels=4,5,6` — cached words, optional level filter
- `GET /episodes/{id}/export.csv` — CSV download

SQLite schema: `episodes(id, url UNIQUE, title, created_at)` and
`words(id, episode_id, word, pinyin, hsk_level, frequency, contexts)` where `contexts`
is a JSON-encoded array stored as text.

`hsk_levels` convention everywhere: comma-separated ints (e.g. `"4,5,6"`) or `"all"`.

## Running it

```bash
pip install -r requirements.txt        # also needs FFmpeg installed system-wide

# CLI
python -m src run "<url>" --model base --hsk-levels 4,5,6 --output ./output

# Web
uvicorn src.app:app --reload           # http://localhost:8000
```

## Conventions

- Match existing style: `from __future__ import annotations`, lowercase-generic type hints
  (`list[str]`, `dict[str, int]`), module-level docstrings.
- Pipeline modules (`audio`, `transcribe`, `extract`, `hsk`) are pure/importable and have
  no CLI or web coupling — keep it that way so both front-ends share one core.
- `static/index.html` is intentionally dependency-free (no npm, no CDN). Keep it vanilla.
- Commit author must be `Claude <noreply@anthropic.com>` or pushes show as Unverified.
- `.claude/` is gitignored.

## Status & roadmap

- **Phase 1 (done):** core pipeline + CLI.
- **Phase 2 (done):** FastAPI web server + single-page UI + SQLite history.
- **Phase 3 (not started):** Anki `.apkg` export (genanki), per-word audio clips,
  configurable `config.yaml`, batch processing, richer definitions.

## Known gaps / things to be aware of

- Definitions are not yet looked up — cards show word, pinyin, HSK level, and context
  sentences only.
- HSK lexicon is a curated subset (~814 words), not the complete official lists.
- No automated tests yet.
- Transcription quality depends heavily on audio clarity and Whisper model size.
