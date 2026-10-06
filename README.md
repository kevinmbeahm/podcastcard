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
- Nothing else to configure: the Whisper model is downloaded automatically on first run

Recommended install (example):

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Usage

Give it a podcast or video URL (anything `yt-dlp` supports):

```bash
python -m src run "https://example.com/episode" --model base --hsk-levels 4,5,6 --output ./output
```

Or start the web UI and paste the URL there:

```bash
uvicorn src.app:app --reload     # http://localhost:8000
```

Options for `run`:

- `--model`: Whisper model size — `tiny`, `base` (default), `small`, `medium`, `large-v2`
- `--device`: `auto` (default; GPU if it works, otherwise CPU), `cpu`, or `cuda`
- `--hsk-levels`: comma-separated levels to show, e.g. `4,5,6`; `0` is words not on any HSK list; default `all`
- `--output`: output directory (default `./output`)

Output in the `--output` folder:

- the downloaded audio (`.mp3`)
- `words.csv` — columns: `word`, `pinyin`, `definition`, `hsk_level`, `frequency`, `contexts` (sentences joined with ` | `)

Planned but not yet implemented: `transcript.vtt`, `phrases.csv`, Anki export, `--min-frequency`, batch mode.

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

## Acknowledgements

HSK word lists come from [complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary)
(MIT), compiled into `data/hsk_words.json` by `scripts/build_hsk_words.py`.

English definitions come from [CC-CEDICT](https://cc-cedict.org), licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The bundled copy is
`data/cedict_ts.u8.gz`.

## License

See LICENSE (if included) or choose an appropriate license for your project.

---

Want me to add a `requirements.txt`, a sample config, or an example CLI runner script next? Reply with which one and I'll add it.
