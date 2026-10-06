"""FastAPI web server for PodcastCard."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
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
from src.transcribe import transcribe
from src.transcript import annotate, format_text, format_vtt

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


def _pipeline(url: str, model: str) -> Iterator[dict]:
    """Download → transcribe → extract → save, yielding progress events.

    Events are ``{"stage", "message"}``; the last is either ``stage == "done"``
    (with ``result``) or ``stage == "error"``.
    """
    yield {"stage": "downloading", "message": "Fetching audio…"}
    title = _fetch_video_title(url)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            audio_path = download_audio(url, tmp)
        except Exception as exc:
            yield {"stage": "error", "message": f"Download failed: {exc}"}
            return

        yield {"stage": "transcribing", "message": "Transcribing audio with Whisper…"}
        try:
            segments = transcribe(audio_path, model_size=model)
        except Exception as exc:
            yield {"stage": "error", "message": f"Transcription failed: {exc}"}
            return

    yield {"stage": "extracting", "message": "Extracting Chinese vocabulary…"}
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
        yield {"stage": "error", "message": f"Extraction failed: {exc}"}
        return

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
