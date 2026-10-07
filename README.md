# PodcastCard — Mandarin podcast transcription & HSK-level word extraction

> Transcribe Mandarin audio, extract words and phrases, and tag them by HSK level for study.

## Overview

PodcastCard takes a link to a Mandarin podcast or video, downloads the audio, transcribes it locally with Whisper, and then:

- gives you the **full transcript**, where any word can be clicked for its pinyin, HSK level, definition and every sentence it appears in
- extracts the **vocabulary**, organised by HSK level, each word keeping the sentences it came from
- exports study material: CSV, **Anki decks**, and the transcript as `.txt` / `.vtt`

It is meant for learners who listen to authentic audio and want to look up exactly the words they didn't understand.

## Features

- Local Mandarin transcription (`faster-whisper`; no API key; falls back to CPU if GPU libraries are missing)
- Click-to-define transcript reader with HSK-level highlighting (web app)
- Word extraction with frequency counts, pinyin, English definitions and HSK levels 1–6, for Simplified *and* Traditional text
- Export: `words.csv`, `transcript.txt`/`.vtt`, and an Anki `.apkg` deck
- Keeps the audio, with a player that follows the transcript and plays any sentence on demand
- Episode history, so earlier analyses reopen instantly

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

### Web app (recommended)

```bash
python -m src serve              # then open http://localhost:8000
```

Paste a podcast or video URL (anything `yt-dlp` supports) and click **Analyze**. When it finishes you get:

- **Transcript tab** — the full time-coded transcript. Words at your selected HSK levels are
  highlighted; click *any* word to see its pinyin, HSK level and definition, plus every sentence in
  the episode where it appears (click a sentence to jump to it). 🔊 reads the word aloud using your
  browser's Chinese voice. Download the transcript as `.txt` or `.vtt`.
- **Full dictionary entry** for the word you click: every reading (行 is *xíng*, *háng* and *héng*) with all its meanings, the Simplified/Traditional form, and a character-by-character breakdown. Phrases that aren't dictionary entries (打篮球) are explained from their parts (*literally: 打 (to hit) + 篮球 (basketball)*), and rare characters are covered too.
- **Audio player** (top of the page, stays pinned while you scroll) — the episode's audio is kept, so you can listen along. Click a timestamp to play from that line; the line being spoken is highlighted and the transcript follows along (switch off with *Follow along*). A ▶ button next to each example sentence plays just that sentence, in the word panel and the Vocabulary tab. *Download audio* saves the file.
- **Vocabulary tab** — the words at the selected HSK levels, grouped by level, each with its
  definition and example sentences. Export as **CSV** or as an **Anki deck**.
- **HSK level chips** (top right) choose which levels are highlighted, listed and exported. Your
  choice is remembered; the default is HSK 4–6.
- **History** (left) reopens earlier episodes without re-processing.

**Where things are stored:** the database (`podcastcard.db`) and the audio files (`podcastcard_audio/`) are created in the folder you start the server from. A 30-minute episode is roughly 40 MB of audio. To keep the audio somewhere else, set `PODCASTCARD_AUDIO_DIR` before starting; to free space, delete files from that folder — the episode keeps its transcript and vocabulary, it just won't have a player.

### Command line

```bash
python -m src run "https://example.com/episode" --model base --hsk-levels 4,5,6 --output ./output --anki
```

Options for `run`:

- `--model`: Whisper model size — `tiny`, `base` (default), `small`, `medium`, `large-v2`
- `--device`: `auto` (default; GPU if it works, otherwise CPU), `cpu`, or `cuda`
- `--hsk-levels`: comma-separated levels to show/export, e.g. `4,5,6`; `0` is words not on any HSK list; default `all`
- `--output`: output directory (default `./output`)
- `--anki`: also write `anki_deck.apkg`

Output in the `--output` folder:

- the downloaded audio (`.mp3`), kept next to the other files
- `transcript.txt` (with `[mm:ss]` markers) and `transcript.vtt` — the full transcript
- `words.csv` — `word`, `pinyin`, `definition`, `hsk_level`, `frequency`, `contexts` (sentences joined with ` | `)
- `anki_deck.apkg` — with `--anki`

### Importing into Anki

In Anki choose **File → Import** and pick the `.apkg`. Each word becomes one note
(word + an example sentence on the front; pinyin, definition, more sentences and HSK level on
the back), tagged `podcastcard` and `HSK<n>`. Notes are keyed by word, so importing another
episode updates words you already have instead of duplicating them.

Planned but not yet implemented: `phrases.csv`, `--min-frequency`, batch mode, audio clips on cards.

## How HSK mapping works

`data/hsk_words.json` maps words to HSK levels 1–6 (Traditional words are looked up through their Simplified form). It is generated by `scripts/build_hsk_words.py` from the official HSK 2.0 lists, using HSK 3.0 levels 1–6 for words the 2.0 lists lack (such as 说 or 天; single characters only up to level 3, since higher ones are mostly parts of compounds like 入). Words in neither list get level 0 ("Non-HSK"), which includes names, slang, loanwords and many everyday compounds.

## Notes & Tips

- The first run of each Whisper model size downloads the model (`tiny` ≈75 MB, `base` ≈145 MB, `small` ≈480 MB, `medium` ≈1.5 GB, `large-v2` ≈3 GB). It uses almost no CPU while downloading, so that wait is normal; the page and terminal show elapsed time so you can tell it is still working.
- Clean audio (good mic, low background noise) greatly improves transcription. `small` is noticeably more accurate than `base`; use `--device cpu` if you don't have a working CUDA setup.
- Most jargon and names show up as *Non-HSK*. Switch that chip on in the web app when you want to see them.

## Contributing

Contributions welcome: bug reports, better phrase extraction, and Anki card templates. Run the tests with `pip install -r requirements-dev.txt && pytest`.

## Acknowledgements

Anki decks are built with [genanki](https://github.com/kerrickstaley/genanki) (MIT).
HSK word lists come from [complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary)
(MIT), compiled into `data/hsk_words.json` by `scripts/build_hsk_words.py`.

English definitions come from [CC-CEDICT](https://cc-cedict.org), licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (`data/cedict_ts.u8.gz`), and, for individual
characters and the Traditional/Simplified mapping, from the Unicode Han Database
(`data/unihan.json.gz`, © Unicode, Inc.; see `data/NOTICE-unihan.txt`).

## License

See LICENSE (if included) or choose an appropriate license for your project.
