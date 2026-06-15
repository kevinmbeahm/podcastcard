"""FastAPI web server for PodcastCard — Phase 2."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow imports from project root when run via uvicorn src.app:app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio import download_audio
from src.transcribe import transcribe
from src.extract import extract_words

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("PODCASTCARD_DB", "./podcastcard.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    url        TEXT    UNIQUE NOT NULL,
    title      TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS words (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    word       TEXT    NOT NULL,
    pinyin     TEXT    NOT NULL,
    hsk_level  INTEGER NOT NULL,
    frequency  INTEGER NOT NULL DEFAULT 1,
    contexts   TEXT    NOT NULL DEFAULT '[]'
);
"""


def _get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    con.commit()
    return con


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PodcastCard", version="2.0.0")

# Mount static files (CSS/JS assets served from /static/*).
# Build the path from this file's location so the server starts regardless of
# the current working directory, and skip the mount if the dir is absent.
_STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hsk_levels(hsk_levels: str) -> list[int] | None:
    """Parse comma-separated HSK level string.  Returns None for 'all'."""
    if not hsk_levels or hsk_levels.strip().lower() == "all":
        return None
    try:
        return [int(x.strip()) for x in hsk_levels.split(",") if x.strip()]
    except ValueError:
        return None


def _words_to_dicts(rows) -> list[dict]:
    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "word": row["word"],
                "pinyin": row["pinyin"],
                "hsk_level": row["hsk_level"],
                "frequency": row["frequency"],
                "contexts": json.loads(row["contexts"]),
            }
        )
    return result


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


def _run_pipeline(url: str, model: str, hsk_levels_str: str):
    """
    Download → transcribe → extract.

    Returns (title, words_list) where words_list is a list of dicts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = download_audio(url, tmp)
        segments = transcribe(audio_path, model_size=model)

    occurrences = extract_words(segments)

    levels_filter = _parse_hsk_levels(hsk_levels_str)
    if levels_filter is not None:
        occurrences = [w for w in occurrences if w.hsk_level in levels_filter]

    # Build word dicts with frequency = number of context sentences
    words = []
    for occ in occurrences:
        words.append(
            {
                "word": occ.word,
                "pinyin": occ.pinyin,
                "hsk_level": occ.hsk_level,
                "frequency": len(occ.contexts),
                "contexts": occ.contexts,
            }
        )

    return words


def _save_episode(url: str, title: str, words: list[dict]) -> int:
    """Persist episode + words to DB.  Returns episode_id."""
    con = _get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()

        # Reuse the existing episode id when the URL was analysed before, so the
        # row's id is stable and its old words can be cleaned up. (INSERT OR
        # REPLACE would allocate a new id and orphan the previous words.)
        existing = con.execute(
            "SELECT id FROM episodes WHERE url = ?", (url,)
        ).fetchone()
        if existing is None:
            cur = con.execute(
                "INSERT INTO episodes (url, title, created_at) VALUES (?, ?, ?)",
                (url, title, now),
            )
            episode_id = cur.lastrowid
        else:
            episode_id = existing["id"]
            con.execute(
                "UPDATE episodes SET title = ?, created_at = ? WHERE id = ?",
                (title, now, episode_id),
            )

        # Delete old words if episode already existed
        con.execute("DELETE FROM words WHERE episode_id = ?", (episode_id,))

        for w in words:
            con.execute(
                "INSERT INTO words (episode_id, word, pinyin, hsk_level, frequency, contexts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    episode_id,
                    w["word"],
                    w["pinyin"],
                    w["hsk_level"],
                    w["frequency"],
                    json.dumps(w["contexts"], ensure_ascii=False),
                ),
            )
        con.commit()
        return episode_id
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page UI."""
    html_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "static",
        "index.html",
    )
    with open(html_path, encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


class AnalyzeRequest(BaseModel):
    url: str
    model: str = "base"
    hsk_levels: str = "all"


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run full pipeline synchronously and return results."""
    title = _fetch_video_title(req.url)
    try:
        words = _run_pipeline(req.url, req.model, req.hsk_levels)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    episode_id = _save_episode(req.url, title, words)
    return JSONResponse({"episode_id": episode_id, "words": words})


@app.get("/episodes")
async def list_episodes():
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
async def get_episode_words(
    episode_id: int, hsk_levels: Optional[str] = Query(default=None)
):
    """Return words for an episode, optionally filtered by HSK level."""
    con = _get_db()
    try:
        ep = con.execute(
            "SELECT id FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if ep is None:
            raise HTTPException(status_code=404, detail="Episode not found")

        levels_filter = _parse_hsk_levels(hsk_levels or "all")
        if levels_filter is not None:
            placeholders = ",".join("?" * len(levels_filter))
            rows = con.execute(
                f"SELECT * FROM words WHERE episode_id = ? AND hsk_level IN ({placeholders}) "
                f"ORDER BY hsk_level, word",
                [episode_id, *levels_filter],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM words WHERE episode_id = ? ORDER BY hsk_level, word",
                (episode_id,),
            ).fetchall()

        return JSONResponse(_words_to_dicts(rows))
    finally:
        con.close()


@app.get("/episodes/{episode_id}/export.csv")
async def export_csv(
    episode_id: int, hsk_levels: Optional[str] = Query(default=None)
):
    """Export episode words as CSV download, optionally filtered by HSK level."""
    con = _get_db()
    try:
        ep = con.execute(
            "SELECT title FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if ep is None:
            raise HTTPException(status_code=404, detail="Episode not found")

        levels_filter = _parse_hsk_levels(hsk_levels or "all")
        if levels_filter is not None:
            placeholders = ",".join("?" * len(levels_filter))
            rows = con.execute(
                "SELECT word, pinyin, hsk_level, frequency, contexts FROM words "
                f"WHERE episode_id = ? AND hsk_level IN ({placeholders}) "
                "ORDER BY hsk_level, word",
                [episode_id, *levels_filter],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT word, pinyin, hsk_level, frequency, contexts FROM words "
                "WHERE episode_id = ? ORDER BY hsk_level, word",
                (episode_id,),
            ).fetchall()
    finally:
        con.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["word", "pinyin", "hsk_level", "frequency", "example_sentence"])
    for row in rows:
        contexts = json.loads(row["contexts"])
        example = contexts[0] if contexts else ""
        writer.writerow(
            [row["word"], row["pinyin"], row["hsk_level"], row["frequency"], example]
        )

    filename = f"episode_{episode_id}.csv"
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/analyze/stream")
async def analyze_stream(
    url: str = Query(...),
    model: str = Query(default="base"),
    hsk_levels: str = Query(default="all"),
):
    """SSE endpoint — streams pipeline progress events."""

    def _event(stage: str, message: str, result=None) -> str:
        payload: dict = {"stage": stage, "message": message}
        if result is not None:
            payload["result"] = result
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _generate():
        try:
            yield _event("downloading", "Fetching audio…")

            title = _fetch_video_title(url)

            with tempfile.TemporaryDirectory() as tmp:
                try:
                    audio_path = download_audio(url, tmp)
                except Exception as exc:
                    yield _event("error", f"Download failed: {exc}")
                    return

                yield _event("transcribing", "Transcribing audio with Whisper…")

                try:
                    segments = transcribe(audio_path, model_size=model)
                except Exception as exc:
                    yield _event("error", f"Transcription failed: {exc}")
                    return

            yield _event("extracting", "Extracting Chinese vocabulary…")

            try:
                occurrences = extract_words(segments)
            except Exception as exc:
                yield _event("error", f"Extraction failed: {exc}")
                return

            levels_filter = _parse_hsk_levels(hsk_levels)
            if levels_filter is not None:
                occurrences = [w for w in occurrences if w.hsk_level in levels_filter]

            words = [
                {
                    "word": occ.word,
                    "pinyin": occ.pinyin,
                    "hsk_level": occ.hsk_level,
                    "frequency": len(occ.contexts),
                    "contexts": occ.contexts,
                }
                for occ in occurrences
            ]

            episode_id = _save_episode(url, title, words)

            yield _event(
                "done",
                f"Found {len(words)} words.",
                result={"episode_id": episode_id, "title": title, "words": words},
            )

        except Exception as exc:
            yield _event("error", f"Unexpected error: {exc}")

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
