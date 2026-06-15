# PodcastCard — Mandarin podcast transcription & HSK-level word extraction

> Transcribe Mandarin audio, extract words and phrases, and tag them by HSK level for study.

## Overview

PodcastCard takes Mandarin podcast audio, generates a time-aligned transcript, then analyzes the transcript to:

- extract words and multi-word phrases
- annotate each token with pinyin, part-of-speech (optional), and HSK level
- rank words and phrases by frequency and contextual usefulness
- export study-ready outputs (CSV, Anki/flashcard-friendly formats, and time-stamped excerpts)

The tool is intended for Mandarin learners and teachers who want targeted vocabulary study from authentic audio. 

## Features

- High-quality Mandarin transcription (local or cloud models supported)
- Word/phrase extraction and frequency counts
- HSK-level mapping (HSK 1–6 and optional extended lists)
- Export: `words.csv`, `phrases.csv`, `transcript.vtt`, and optional Anki `.apkg` or CSV deck
- Configurable filters: minimum frequency, part-of-speech filters, phrase length, context window
- Batch processing for multiple episodes

## Quick Start

Prerequisites:

- Python 3.10+
- FFmpeg (for audio decoding)
- A speech model (e.g., Whisper family or cloud ASR key) — configurable in settings

Recommended install (example):

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Usage

Basic CLI example:

```bash
podcastcard transcribe \\
  --input episode01.mp3 \\
  --output out/episode01 \\
  --lang zh \\
  --model small \\
  --hsk-levels 1,2,3
```

Options (common):

- `--input`: path to audio file (mp3, m4a, wav, etc.) or directory for batch
- `--output`: output directory
- `--lang`: language code (default `zh`)
- `--model`: transcription model or `auto` (local or cloud)
- `--hsk-levels`: comma-separated HSK levels to emit or `all`
- `--min-frequency`: filter words with frequency lower than this
- `--export-anki`: produce an Anki-compatible deck (CSV or `.apkg`)

Example output files created in the `--output` folder:

- `transcript.vtt` — time-coded transcript
- `words.csv` — columns: `word`, `pinyin`, `hsk_level`, `frequency`, `example_context`, `first_timestamp`
- `phrases.csv` — extracted useful multi-word phrases with counts and timestamps
- `anki_deck.csv` or `anki_deck.apkg` — flashcard-ready output

## How HSK mapping works

PodcastCard includes a built-in HSK lexicon that maps common words and phrases to HSK levels 1–6. Behavior is configurable:

- default mapping uses official HSK lists (and community extensions if enabled)
- unknown words get `hsk_level = 0` (unlisted)
- you can provide a custom mapping CSV for institutional vocab lists

## Configuration

Configuration can be provided via a YAML/JSON file or CLI flags. Typical config options:

- `transcription.model` (string) — model name or API key
- `analysis.min_frequency` (int)
- `analysis.pos_filter` (list)
- `output.formats` (list)
- `hsk.mapping_path` (path)

## Advanced usage

- Batch mode: pass a folder to `--input` to process many episodes
- SRS integration: export Anki-ready decks with sentence context and audio clips
- Timestamped examples: include short audio clips per word for pronunciation practice

## Notes & Tips

- Clean audio (good mic, low background noise) greatly improves transcription and word extraction quality.
- For best results with learner-focused extraction, filter out proper nouns and high-frequency function words using `--min-frequency` and `--pos-filter`.

## Contributing

Contributions welcome: bug reports, additional HSK lists, improved phrase extraction heuristics, and Anki export templates.

## License

See LICENSE (if included) or choose an appropriate license for your project.

---

Want me to add a `requirements.txt`, a sample config, or an example CLI runner script next? Reply with which one and I'll add it.
