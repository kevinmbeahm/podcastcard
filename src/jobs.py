"""A background queue for analysing several videos, one after another.

Jobs live in SQLite, so the queue survives closing the browser tab and restarting the
server. A single worker runs them in order (transcription is heavy, so there is no point
running two at once). The pipeline that analyses one video is passed in, which keeps this
module independent of the web app and easy to test.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, Iterator
from urllib.parse import urlparse

log = logging.getLogger("uvicorn.error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL,
    model       TEXT    NOT NULL DEFAULT 'base',
    status      TEXT    NOT NULL DEFAULT 'queued',
    stage       TEXT    NOT NULL DEFAULT '',
    message     TEXT    NOT NULL DEFAULT '',
    progress    REAL,
    elapsed     INTEGER NOT NULL DEFAULT 0,
    title       TEXT    NOT NULL DEFAULT '',
    episode_id  INTEGER,
    created_at  TEXT    NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, id);
"""

QUEUED, RUNNING, DONE, SKIPPED, ERROR, CANCELLED = (
    "queued",
    "running",
    "done",
    "skipped",
    "error",
    "cancelled",
)
FINISHED = (DONE, SKIPPED, ERROR, CANCELLED)

MAX_BATCH = 200  # links per request: a guard against pasting something enormous
_WRITE_INTERVAL = 0.5  # seconds between progress writes to the database
_UPDATABLE = {"status", "stage", "message", "progress", "elapsed", "title", "episode_id", "started_at", "finished_at"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _close(events: Iterator[dict]) -> None:
    """Close a generator (which stops the work behind it); plain iterators have nothing to close."""
    close = getattr(events, "close", None)
    if close:
        close()


def _is_web_link(token: str) -> bool:
    parsed = urlparse(token)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def parse_urls(text: str | Iterable[str]) -> tuple[list[str], list[dict]]:
    """Pull links out of pasted text (or a list of such pieces).

    One link per line is the normal case, but several per line work too, and words around a
    link ("Episode 3 – https://…") are ignored. Blank lines and lines starting with ``#`` are
    skipped. Only http(s) links are accepted: they are handed to yt-dlp, which would also
    happily read local files or other schemes. A line with no usable link is reported, once,
    as ``{"line": ..., "reason": ...}``. Returns ``(links, rejected)``.
    """
    pieces = [text] if isinstance(text, str) else list(text)
    links: list[str] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for piece in pieces:
        for line in piece.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            found = [token for token in line.split() if _is_web_link(token)]
            if not found:
                rejected.append(
                    {"line": line, "reason": "Not a web link (it has to start with http:// or https://)"}
                )
            for token in found:
                if token not in seen:
                    seen.add(token)
                    links.append(token)
    return links, rejected


class JobQueue:
    """Queue of analysis jobs with one background worker.

    ``connect`` returns a new SQLite connection (``row_factory = sqlite3.Row``, schema
    present). ``pipeline(url, model)`` yields progress events and ends with a ``"done"``
    event (with ``result: {episode_id, title}``) or an ``"error"`` event; closing it must
    stop the work.
    """

    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        pipeline: Callable[[str, str], Iterator[dict]],
    ) -> None:
        self._connect = connect
        self._pipeline = pipeline
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancelling: set[int] = set()
        self._cancelling_lock = threading.Lock()

    # ------------------------------------------------------------------ database helpers

    def _read(self, sql: str, params: tuple = ()) -> list[dict]:
        con = self._connect()
        try:
            return [dict(row) for row in con.execute(sql, params).fetchall()]
        finally:
            con.close()

    def _write(self, sql: str, params: tuple = ()) -> int:
        con = self._connect()
        try:
            cursor = con.execute(sql, params)
            con.commit()
            return cursor.rowcount
        finally:
            con.close()

    def _update(self, job_id: int, **fields) -> None:
        assert fields.keys() <= _UPDATABLE, fields.keys()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self._write(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))

    # ------------------------------------------------------------------ public API

    def get(self, job_id: int) -> dict | None:
        rows = self._read("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0] if rows else None

    def list_jobs(self, limit: int = 500) -> list[dict]:
        """Every job still on the list, in the order they are (or were) processed."""
        return self._read("SELECT * FROM jobs ORDER BY id LIMIT ?", (limit,))

    def enqueue(self, urls: list[str], model: str = "base", skip_existing: bool = True) -> dict:
        """Add links to the queue. Returns ``{"jobs": [...], "rejected": [...]}``.

        A link that is already waiting or running is not added twice. With
        ``skip_existing`` a link whose episode has been analysed before is recorded as
        ``skipped`` (pointing at that episode) instead of being processed again.
        """
        created: list[int] = []
        rejected: list[dict] = []
        con = self._connect()
        try:
            active = {
                row["url"]
                for row in con.execute("SELECT url FROM jobs WHERE status IN (?, ?)", (QUEUED, RUNNING))
            }
            for url in urls:
                if url in active:
                    rejected.append({"line": url, "reason": "Already in the queue"})
                    continue
                episode = (
                    con.execute("SELECT id, title FROM episodes WHERE url = ?", (url,)).fetchone()
                    if skip_existing
                    else None
                )
                if episode:
                    cursor = con.execute(
                        "INSERT INTO jobs (url, model, status, stage, message, progress, title, episode_id, "
                        "created_at, finished_at) VALUES (?, ?, ?, 'done', 'Already analyzed', 1.0, ?, ?, ?, ?)",
                        (url, model, SKIPPED, episode["title"], episode["id"], _now(), _now()),
                    )
                else:
                    cursor = con.execute(
                        "INSERT INTO jobs (url, model, status, created_at) VALUES (?, ?, ?, ?)",
                        (url, model, QUEUED, _now()),
                    )
                    active.add(url)
                created.append(cursor.lastrowid)
            con.commit()
        finally:
            con.close()
        self._wake.set()
        jobs = [job for job in (self.get(job_id) for job_id in created) if job]
        return {"jobs": jobs, "rejected": rejected}

    def cancel(self, job_id: int) -> dict | None:
        """Stop a job: a waiting one is dropped at once, a running one within a couple of seconds."""
        job = self.get(job_id)
        if job is None:
            return None
        if job["status"] == QUEUED:
            changed = self._write(
                "UPDATE jobs SET status = ?, message = 'Cancelled', finished_at = ? WHERE id = ? AND status = ?",
                (CANCELLED, _now(), job_id, QUEUED),
            )
            if changed:
                return self.get(job_id)
            job = self.get(job_id)  # the worker picked it up just now: fall through
        if job and job["status"] == RUNNING:
            with self._cancelling_lock:
                self._cancelling.add(job_id)
            self._update(job_id, message="Cancelling…")
        return self.get(job_id)

    def retry(self, job_id: int) -> dict | None:
        """Put a failed or cancelled job back at the end of the queue.

        Raises ``ValueError`` for a job in any other state, ``None`` if there is no such job.
        """
        job = self.get(job_id)
        if job is None:
            return None
        if job["status"] not in (ERROR, CANCELLED):
            raise ValueError(f"A job that is {job['status']} cannot be retried")
        self._write(
            "UPDATE jobs SET status = ?, stage = '', message = '', progress = NULL, elapsed = 0, "
            "started_at = NULL, finished_at = NULL WHERE id = ?",
            (QUEUED, job_id),
        )
        self._wake.set()
        return self.get(job_id)

    def clear_finished(self) -> int:
        """Remove finished jobs (done, skipped, failed, cancelled) from the list."""
        marks = ", ".join("?" * len(FINISHED))
        return self._write(f"DELETE FROM jobs WHERE status IN ({marks})", FINISHED)

    # ------------------------------------------------------------------ the worker

    def start(self) -> None:
        """Resume interrupted work and start the background worker."""
        self._write(
            "UPDATE jobs SET status = ?, stage = '', progress = NULL, "
            "message = 'Interrupted by a restart; starting again' WHERE status = ?",
            (QUEUED, RUNNING),
        )
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-queue")
        self._thread.start()
        self._wake.set()

    def stop(self, wait: bool = False, timeout: float = 30.0) -> None:
        """Ask the worker to stop after the job it is on. ``wait`` blocks until it has."""
        self._stop.set()
        self._wake.set()
        if wait and self._thread:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                processed = self.run_pending()
            except Exception:
                log.exception("The job queue hit an unexpected error")
                processed = 0
            if processed == 0:
                self._wake.wait(timeout=5)
                self._wake.clear()

    def run_pending(self) -> int:
        """Process queued jobs, oldest first, until none are left. Returns how many ran."""
        count = 0
        while not self._stop.is_set():
            job = self._claim_next()
            if job is None:
                break
            self._run_one(job)
            count += 1
        return count

    def _claim_next(self) -> dict | None:
        while True:
            rows = self._read("SELECT * FROM jobs WHERE status = ? ORDER BY id LIMIT 1", (QUEUED,))
            if not rows:
                return None
            job = rows[0]
            claimed = self._write(
                "UPDATE jobs SET status = ?, stage = 'downloading', message = 'Starting…', progress = NULL, "
                "elapsed = 0, started_at = ?, finished_at = NULL WHERE id = ? AND status = ?",
                (RUNNING, _now(), job["id"], QUEUED),
            )
            if claimed:
                return job
            # otherwise it was cancelled between the SELECT and the UPDATE: look at the next one

    def _run_one(self, job: dict) -> None:
        job_id = job["id"]
        log.info("Job %s: %s", job_id, job["url"])
        events = self._pipeline(job["url"], job["model"])
        last_stage, last_write = "", 0.0
        try:
            for event in events:
                stage = event.get("stage", "")
                if stage == "done":
                    result = event.get("result", {})
                    self._finish(
                        job_id,
                        DONE,
                        stage="done",
                        message=event.get("message", ""),
                        progress=1.0,
                        episode_id=result.get("episode_id"),
                        title=result.get("title", ""),
                    )
                    return
                if stage == "error":
                    self._finish(job_id, ERROR, stage="error", message=event.get("message", "Failed"))
                    return
                if self._should_cancel(job_id):
                    _close(events)
                    self._finish(job_id, CANCELLED, message="Cancelled")
                    return
                now = time.monotonic()
                if stage != last_stage or now - last_write >= _WRITE_INTERVAL:
                    fields = {
                        "stage": stage,
                        "message": event.get("message", ""),
                        "progress": event.get("progress"),
                        "elapsed": int(event.get("elapsed") or 0),
                    }
                    if event.get("title"):
                        fields["title"] = event["title"]
                    self._update(job_id, **fields)
                    last_stage, last_write = stage, now
            self._finish(job_id, ERROR, stage="error", message="The analysis ended without a result")
        except Exception as exc:
            log.exception("Job %s failed unexpectedly", job_id)
            self._finish(job_id, ERROR, stage="error", message=f"Unexpected error: {exc}")
        finally:
            _close(events)
            with self._cancelling_lock:
                self._cancelling.discard(job_id)

    def _should_cancel(self, job_id: int) -> bool:
        with self._cancelling_lock:
            return job_id in self._cancelling

    def _finish(self, job_id: int, status: str, **fields) -> None:
        self._update(job_id, status=status, finished_at=_now(), **fields)
