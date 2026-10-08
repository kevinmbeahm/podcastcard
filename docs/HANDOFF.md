# PodcastCard — Handoff

Snapshot of where the project stands. `CLAUDE.md` is the architecture/API reference; this file adds status, history, decisions, open questions and next steps.

## 1. State

- Branch `claude/intelligent-wozniak-svoqmh`. PR #1 was merged into `main` early (Phase 2 only). Everything after it (dictionary, HSK lists, reader, audio, process isolation, batch queue) is **not yet on `main`**. Open a new PR from this branch when wanted.
- `main` also holds the user's own commits (`72890c0`, `67717e8`): `setup.bat`, `run-web.bat`, `run-cli.bat`, a `.gitignore`, and small `app.py` fixes (static dir path, filtered CSV). The branch already had equivalents of the `app.py` fixes; the merge (`55aeefd`) kept the branch's `app.py`.
- 167 tests pass (`pip install -r requirements-dev.txt && pytest`).
- **Never verified end-to-end in the sandbox**: real download + Whisper transcription (YouTube, huggingface.co blocked, no FFmpeg). The user ran those paths locally.

## 2. User environment

- Windows, **ARM machine** (per `setup.bat`): faster-whisper/ctranslate2/av only ship x64 wheels, so the venv must use an x64 Python (runs emulated). NVIDIA CUDA libs absent -> CPU int8 fallback (`--device cpu`).
- Start with `setup.bat`, then `run-web.bat` (uvicorn `--reload`, http://localhost:8000) or `run-cli.bat`. `python -m src serve` also works (`uvicorn` was not on PATH).
- Needs FFmpeg on PATH (`winget install Gyan.FFmpeg`).
- Heads-up: `run-web.bat` uses `--reload`; combined with spawned Whisper children and the queue worker thread this restarts the server on any file change. Drop `--reload` if odd restarts appear.

## 3. Feature history (commits)

| Commit | What |
|---|---|
| 0817402 | Phase 1 pipeline + CLI |
| 0b8174d, eb44cfd | Phase 2 FastAPI + SPA + SQLite history |
| 8ac7db5 | CC-CEDICT definitions |
| a656057 | Full HSK 1–6 lists |
| c035afc, bf3349e, 37be041, d9884c1 | CLI subcommand fix, FFmpeg check, CUDA->CPU fallback, `serve` |
| 660dfd4 | Transcript reader, click-to-define, Vocabulary tab, Anki export |
| 4370d7e, 3680c1f | Live progress/heartbeats; interrupted model download no longer looks "cached" |
| 379602d | Each transcription in its own spawned process |
| 60054bf | Stored audio + player |
| 34c9723 | Fuller dictionary (Unihan, Traditional input, cross-refs) |
| fecb2b5 | Stop bare characters (入) appearing as HSK 4–6 |
| b7b53ac | Batch queue (CLI multiple URLs/`--file`; web queue) |

## 4. Key design decisions

- Everything is stored; HSK level chips are only a view filter.
- Tokenisation + lexicon computed once and stored; episodes refresh from stored text when `DICT_VERSION` increases (no audio/Whisper needed). **Bump `DICT_VERSION` when segmentation, filtering or dictionary output changes.**
- Traditional text: segment the Simplified form, cut the original text at the same positions (keeps script).
- One analysis at a time (`_analysis_lock`); queue worker is a normal caller. Cancel = flag + closing the generator.
- Playlist/channel links are refused (`PlaylistLinkError`) to avoid silently queueing days of work.
- HSK data: 2.0 first, 3.0 1–6 fallback; bare 3.0 characters only up to level 3.

## 5. Known limitations

See `CLAUDE.md` "Known gaps". Highlights: no per-context definitions; no Chinese–Chinese dictionary (idioms); Non-HSK noisy for trivial compounds (一个, 一下); job title known only once transcription starts; queue progress polled each second; audio never deleted and no delete-episode; HSK 3.0 7–9 band absent.

## 6. Unconfirmed

- Whether process isolation fixed the "works once, then hangs on Loading model" report (user never confirmed).
- Whether the 入 fix satisfied the user.
- yt-dlp `noplaylist`/playlist detection tested only with stand-ins.
- Browser UI checked with Playwright on seeded data; those scripts were not committed.

## 7. Open questions for the user

1. Want a Chinese–Chinese dictionary (for idioms)?
2. Hide single-character tokens in Vocabulary?
3. Android: PWA/share-target interest? (Feasibility discussed, no code.)
4. Playlist expansion with a cap/confirmation?
5. Delete-episode/audio feature?

## 8. Suggested next steps

1. Confirm hang fix and batch queue on the user's machine; open a PR to `main`.
2. Delete episode / audio.
3. Reduce Non-HSK noise (treat compounds of known words as known).
4. Per-word audio clips on Anki cards; `config.yaml`; `--standard` HSK option.
5. Playlist expansion with confirmation; push-based queue progress.
6. Context-aware definitions.

## 9. Data provenance / regeneration

- `data/hsk_words.json`: `python scripts/build_hsk_words.py` (drkameleon/complete-hsk-vocabulary, MIT).
- `data/unihan.json.gz`: `python scripts/build_unihan.py` (needs unicode.org; blocked in the Claude sandbox).
- `data/cedict_ts.u8.gz`: no build script; from TeaPearce/chinese-english-dictionary `data/cedict_ts_june2026.u8` on GitHub, gzip -9.

## 10. Licensing action items

No LICENSE file yet. CC-CEDICT is CC BY-SA 4.0 (attribution + share-alike). Unihan terms live at unicode.org/copyright.html (text not bundled). Add a LICENSE and a combined third-party notice before publishing.

## 11. Sandbox notes

Network proxy blocks youtube.com, huggingface.co, unicode.org; GitHub raw/PyPI reachable. No FFmpeg. jieba needs a venv with a recent setuptools. Don't `pkill -f` (kills your own shell).
