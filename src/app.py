"""FastAPI web server for PodcastCard."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional, TypeVar
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow imports from project root when run via uvicorn src.app:app
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.anki import write_apkg
from src.audio import download_audio
from src.extract import Segment, extract_words
from src.transcribe import loading_message
from src.transcribe import transcribe_isolated as transcribe
from src.transcript import annotate, format_text, format_vtt

# uvicorn's logger, so progress and errors show up in the terminal running the server
log = logging.getLogger("uvicorn.error")

# How often a long-running stage re-sends its status, so the page can show it is alive.
HEARTBEAT_SECONDS = 2.0

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("PODCASTCARD_DB", "./podcastcard.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    url        TEXT    UNIQUE NOT NULL,
    title      TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    lexicon    TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS words (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    word       TEXT    NOT NULL,
    pinyin     TEXT    NOT NULL,
    definition TEXT    NOT NULL DEFAULT '',
    hsk_level  INTEGER NOT NULL,
    frequency  INTEGER NOT NULL DEFAULT 1,
    contexts   TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    idx        INTEGER NOT NULL,
    start      REAL    NOT NULL,
    end        REAL    NOT NULL,
    text       TEXT    NOT NULL,
    tokens     TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS segments_episode ON segments(episode_id, idx);
"""

# Columns added after the first release: (table, column, definition)
_MIGRATIONS = [
    ("words", "definition", "TEXT NOT NULL DEFAULT ''"),
    ("episodes", "lexicon", "TEXT NOT NULL DEFAULT '{}'"),
]


def _get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    for table, column, ddl in _MIGRATIONS:
        cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    con.commit()
    return con


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PodcastCard", version="2.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hsk_levels(hsk_levels: Optional[str]) -> list[int] | None:
    """Parse a comma-separated HSK level string.  Returns None for 'all'."""
    if not hsk_levels or hsk_levels.strip().lower() == "all":
        return None
    try:
        return [int(x.strip()) for x in hsk_levels.split(",") if x.strip()]
    except ValueError:
        return None


def _words_to_dicts(rows) -> list[dict]:
    return [
        {
            "id": row["id"],
            "word": row["word"],
            "pinyin": row["pinyin"],
            "definition": row["definition"],
            "hsk_level": row["hsk_level"],
            "frequency": row["frequency"],
            "contexts": json.loads(row["contexts"]),
        }
        for row in rows
    ]


def _require_episode(con: sqlite3.Connection, episode_id: int) -> sqlite3.Row:
    ep = con.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return ep


def _select_words(con: sqlite3.Connection, episode_id: int, hsk_levels: Optional[str]) -> list[dict]:
    """An episode's words, optionally restricted to some HSK levels."""
    levels = _parse_hsk_levels(hsk_levels)
    sql = "SELECT * FROM words WHERE episode_id = ?"
    params: list = [episode_id]
    if levels is not None:
        sql += f" AND hsk_level IN ({','.join('?' * len(levels))})"
        params += levels
    rows = con.execute(sql + " ORDER BY hsk_level = 0, hsk_level, word", params).fetchall()
    return _words_to_dicts(rows)


def _load_segments(con: sqlite3.Connection, episode_id: int) -> list[Segment]:
    rows = con.execute(
        "SELECT start, end, text FROM segments WHERE episode_id = ? ORDER BY idx", (episode_id,)
    ).fetchall()
    return [Segment(r["start"], r["end"], r["text"]) for r in rows]


def _download_headers(episode_id: int, title: str, ext: str) -> dict[str, str]:
    """Content-Disposition with an ASCII fallback plus the real (Unicode) title."""
    pretty = quote(f"{title}.{ext}")
    return {
        "Content-Disposition": f"attachment; filename=\"episode_{episode_id}.{ext}\"; "
        f"filename*=UTF-8''{pretty}"
    }


def _fetch_video_title(url: str) -> str:
    """Try to get the video/podcast title from yt-dlp without downloading."""
    try:
        import yt_dlp

        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title") or url
    except Exception:
        return url


def _save_episode(
    url: str,
    title: str,
    words: list[dict],
    segments: list[dict] | None = None,
    lexicon: dict | None = None,
) -> int:
    """Persist (or replace) an episode with its words and transcript. Returns its id."""
    con = _get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        lexicon_json = json.dumps(lexicon or {}, ensure_ascii=False)
        row = con.execute("SELECT id FROM episodes WHERE url = ?", (url,)).fetchone()
        if row:
            episode_id = row["id"]
            con.execute(
                "UPDATE episodes SET title = ?, created_at = ?, lexicon = ? WHERE id = ?",
                (title, now, lexicon_json, episode_id),
            )
            con.execute("DELETE FROM words WHERE episode_id = ?", (episode_id,))
            con.execute("DELETE FROM segments WHERE episode_id = ?", (episode_id,))
        else:
            cur = con.execute(
                "INSERT INTO episodes (url, title, created_at, lexicon) VALUES (?, ?, ?, ?)",
                (url, title, now, lexicon_json),
            )
            episode_id = cur.lastrowid

        con.executemany(
            "INSERT INTO words (episode_id, word, pinyin, definition, hsk_level, frequency, contexts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    episode_id,
                    w["word"],
                    w["pinyin"],
                    w["definition"],
                    w["hsk_level"],
                    w["frequency"],
                    json.dumps(w["contexts"], ensure_ascii=False),
                )
                for w in words
            ],
        )
        con.executemany(
            "INSERT INTO segments (episode_id, idx, start, end, text, tokens) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (episode_id, i, s["start"], s["end"], s["text"], json.dumps(s["tokens"], ensure_ascii=False))
                for i, s in enumerate(segments or [])
            ],
        )
        con.commit()
        return episode_id
    finally:
        con.close()


def _fmt_clock(seconds: float) -> str:
    m, sec = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _run_with_progress(
    work: Callable[[Callable[[Optional[float], Optional[str]], None]], T],
    stage: str,
    message: str,
    started: float,
) -> Generator[dict, None, T]:
    """Run blocking ``work(report)`` in a thread, yielding progress events.

    ``report(fraction, message)`` may be called from the worker at any time. An event is
    yielded for each report and, if nothing is reported, every ``HEARTBEAT_SECONDS`` --
    so slow, silent steps (a model download, FFmpeg) still show elapsed time. Returns
    work's result, or re-raises its exception.
    """
    updates: queue.Queue = queue.Queue()
    cancelled = threading.Event()  # set when the client goes away, so work can stop

    def report(fraction: Optional[float] = None, text: Optional[str] = None) -> None:
        updates.put(("progress", fraction, text))

    report.cancelled = cancelled  # type: ignore[attr-defined]

    def target() -> None:
        try:
            updates.put(("done", work(report), None))
        except BaseException as exc:  # handed back to the caller's thread
            updates.put(("error", exc, None))

    def event(fraction: Optional[float]) -> dict:
        return {
            "stage": stage,
            "message": message,
            "progress": fraction,
            "elapsed": round(time.monotonic() - started),
        }

    threading.Thread(target=target, daemon=True).start()

    fraction: Optional[float] = None
    try:
        yield event(None)  # announce the stage straight away, don't wait for the first heartbeat
        while True:
            try:
                kind, value, text = updates.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                kind = "heartbeat"
            if kind == "done":
                return value
            if kind == "error":
                raise value
            if kind == "progress":
                fraction = value  # None means "unknown": show the animated bar again
                message = text or message
            yield event(fraction)
    finally:
        # Also runs if the generator is closed early (client disconnected): tell the
        # worker to stop instead of letting it keep burning CPU.
        cancelled.set()


def _throttled(report: Callable[..., None], step: float = 0.01) -> Callable[[float, str], None]:
    """Only forward a progress report when it has moved by at least *step*."""
    last = [-1.0]

    def inner(fraction: float, text: str) -> None:
        if fraction >= 1.0 or fraction - last[0] >= step:
            last[0] = fraction
            report(fraction, text)

    return inner


def _pipeline(url: str, model: str) -> Iterator[dict]:
    """Download → transcribe → extract → save, yielding progress events.

    Events are ``{"stage", "message", "progress"?, "elapsed"?}``; the last is either
    ``stage == "done"`` (with ``result``) or ``stage == "error"``.
    """
    started = time.monotonic()
    log.info("Analyzing %s (model=%s)", url, model)
    yield {"stage": "downloading", "message": "Fetching audio…", "elapsed": 0}

    with tempfile.TemporaryDirectory() as tmp:
        try:

            def download(report):
                progress = _throttled(report)
                # Also a network call, so it runs here where the heartbeat covers it.
                title = _fetch_video_title(url)

                def on_download(fraction: float) -> None:
                    text = (
                        "Converting audio to mp3…"
                        if fraction >= 1.0
                        else f"Downloading audio… {fraction:.0%}"
                    )
                    progress(fraction, text)

                return title, download_audio(url, tmp, on_progress=on_download)

            title, audio_path = yield from _run_with_progress(
                download, "downloading", "Looking up the video…", started
            )
        except Exception as exc:
            log.exception("Download failed for %s", url)
            yield {"stage": "error", "message": f"Download failed: {exc}"}
            return

        loading = loading_message(model)
        log.info(loading)

        try:

            def run_whisper(report):
                progress = _throttled(report)

                def on_status(text: str, fraction: Optional[float] = None) -> None:
                    if fraction is None:  # stage changes only; download ticks would flood the log
                        log.info(text)
                    report(fraction, text)

                def on_transcribe(done: float, total: float) -> None:
                    progress(
                        done / total,
                        f"Transcribing… {done / total:.0%} "
                        f"({_fmt_clock(done)} of {_fmt_clock(total)})",
                    )

                return transcribe(
                    audio_path,
                    model_size=model,
                    on_progress=on_transcribe,
                    on_status=on_status,
                    should_cancel=report.cancelled.is_set,
                )

            segments = yield from _run_with_progress(
                run_whisper, "transcribing", loading, started
            )
        except Exception as exc:
            log.exception("Transcription failed for %s", url)
            yield {"stage": "error", "message": f"Transcription failed: {exc}"}
            return

    yield {"stage": "extracting", "message": "Extracting Chinese vocabulary…", "elapsed": round(time.monotonic() - started)}
    try:
        # Every word is stored: the HSK filter is applied when viewing/exporting.
        words = [
            {
                "word": occ.word,
                "pinyin": occ.pinyin,
                "definition": occ.definition,
                "hsk_level": occ.hsk_level,
                "frequency": len(occ.contexts),
                "contexts": occ.contexts,
            }
            for occ in extract_words(segments)
        ]
        annotated, lexicon = annotate(segments)
        episode_id = _save_episode(url, title, words, annotated, lexicon)
    except Exception as exc:
        log.exception("Extraction failed for %s", url)
        yield {"stage": "error", "message": f"Extraction failed: {exc}"}
        return

    log.info("Finished %s in %ss: %d words, %d segments", url, round(time.monotonic() - started), len(words), len(segments))
    yield {
        "stage": "done",
        "message": f"Found {len(words)} words in {len(segments)} transcript segments.",
        "result": {"episode_id": episode_id, "title": title},
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page UI."""
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"))


class AnalyzeRequest(BaseModel):
    url: str
    model: str = "base"
    hsk_levels: str = "all"


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run the full pipeline and return the episode id and its (filtered) words."""
    final: dict = {}
    for event in _pipeline(req.url, req.model):
        final = event
    if final.get("stage") != "done":
        raise HTTPException(status_code=500, detail=final.get("message", "Analysis failed"))
    episode_id = final["result"]["episode_id"]
    con = _get_db()
    try:
        words = _select_words(con, episode_id, req.hsk_levels)
    finally:
        con.close()
    return JSONResponse({"episode_id": episode_id, "words": words})


@app.get("/analyze/stream")
def analyze_stream(url: str = Query(...), model: str = Query(default="base")):
    """SSE endpoint — streams pipeline progress events."""

    def _generate():
        try:
            for event in _pipeline(url, model):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # last-resort guard so the client always gets an end event
            err = {"stage": "error", "message": f"Unexpected error: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/episodes")
def list_episodes():
    """Return list of all episodes."""
    con = _get_db()
    try:
        rows = con.execute(
            "SELECT id, url, title, created_at FROM episodes ORDER BY id DESC"
        ).fetchall()
        return JSONResponse([dict(r) for r in rows])
    finally:
        con.close()


@app.get("/episodes/{episode_id}/words")
def get_episode_words(episode_id: int, hsk_levels: Optional[str] = Query(default=None)):
    """Return words for an episode, optionally filtered by HSK level."""
    con = _get_db()
    try:
        _require_episode(con, episode_id)
        return JSONResponse(_select_words(con, episode_id, hsk_levels))
    finally:
        con.close()


@app.get("/episodes/{episode_id}/transcript")
def get_transcript(episode_id: int):
    """The full transcript, tokenised, plus a lexicon for looking up any word in it."""
    con = _get_db()
    try:
        ep = _require_episode(con, episode_id)
        rows = con.execute(
            "SELECT start, end, text, tokens FROM segments WHERE episode_id = ? ORDER BY idx",
            (episode_id,),
        ).fetchall()
        return JSONResponse(
            {
                "episode": {"id": ep["id"], "url": ep["url"], "title": ep["title"]},
                "segments": [
                    {
                        "start": r["start"],
                        "end": r["end"],
                        "text": r["text"],
                        "tokens": json.loads(r["tokens"]),
                    }
                    for r in rows
                ],
                "lexicon": json.loads(ep["lexicon"]),
            }
        )
    finally:
        con.close()


@app.get("/episodes/{episode_id}/transcript.vtt")
def export_vtt(episode_id: int):
    """Time-coded transcript as a WebVTT file."""
    con = _get_db()
    try:
        ep = _require_episode(con, episode_id)
        segments = _load_segments(con, episode_id)
    finally:
        con.close()
    return Response(
        format_vtt(segments),
        media_type="text/vtt; charset=utf-8",
        headers=_download_headers(episode_id, ep["title"], "vtt"),
    )


@app.get("/episodes/{episode_id}/transcript.txt")
def export_transcript_text(episode_id: int):
    """Plain-text transcript with [mm:ss] markers."""
    con = _get_db()
    try:
        ep = _require_episode(con, episode_id)
        segments = _load_segments(con, episode_id)
    finally:
        con.close()
    return Response(
        format_text(segments),
        media_type="text/plain; charset=utf-8",
        headers=_download_headers(episode_id, ep["title"], "txt"),
    )


@app.get("/episodes/{episode_id}/export.csv")
def export_csv(episode_id: int, hsk_levels: Optional[str] = Query(default=None)):
    """Export episode words as a CSV download (optionally only some HSK levels)."""
    con = _get_db()
    try:
        ep = _require_episode(con, episode_id)
        words = _select_words(con, episode_id, hsk_levels)
    finally:
        con.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["word", "pinyin", "definition", "hsk_level", "frequency", "example_sentence"])
    for w in words:
        example = w["contexts"][0] if w["contexts"] else ""
        writer.writerow(
            [w["word"], w["pinyin"], w["definition"], w["hsk_level"], w["frequency"], example]
        )
    return Response(
        buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers=_download_headers(episode_id, ep["title"], "csv"),
    )


@app.get("/episodes/{episode_id}/anki.apkg")
def export_anki(episode_id: int, hsk_levels: Optional[str] = Query(default=None)):
    """Export episode words as an importable Anki deck (optionally only some HSK levels)."""
    con = _get_db()
    try:
        ep = _require_episode(con, episode_id)
        words = _select_words(con, episode_id, hsk_levels)
    finally:
        con.close()
    if not words:
        raise HTTPException(status_code=404, detail="No words match the selected HSK levels")

    # "::" would make Anki create sub-decks, so keep it out of the episode title.
    deck_name = f"PodcastCard::{ep['title'].replace('::', ':')}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deck.apkg"
        write_apkg(words, path, deck_name, source=ep["title"])
        data = path.read_bytes()
    return Response(
        data,
        media_type="application/octet-stream",
        headers=_download_headers(episode_id, ep["title"], "apkg"),
    )
