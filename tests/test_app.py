"""Web API tests (download/transcribe are stubbed; uses a temporary database)."""

import io
import json
import os
import sqlite3
import tempfile
import zipfile

import pytest
from fastapi.testclient import TestClient

import src.app as app_module
from src.extract import Segment

URL = "https://example.com/ep1"
SEGMENTS = [
    Segment(0.0, 3.5, "我喜欢听中文播客，学习新的词汇。"),
    Segment(3.5, 6.0, "播客很有意思。"),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(
        app_module, "download_audio", lambda url, out, on_progress=None: f"{out}/x.mp3"
    )
    monkeypatch.setattr(
        app_module, "transcribe", lambda path, model_size="base", on_progress=None: SEGMENTS
    )
    monkeypatch.setattr(app_module, "loading_message", lambda model: f"Loading {model}")
    monkeypatch.setattr(app_module, "_fetch_video_title", lambda url: "测试 Episode: one")
    return TestClient(app_module.app)


def analyze(client, url=URL, **body):
    r = client.post("/analyze", json={"url": url, **body})
    assert r.status_code == 200, r.text
    return r.json()["episode_id"]


def test_analyze_stores_every_word_but_returns_only_requested_levels(client):
    r = client.post("/analyze", json={"url": URL, "hsk_levels": "0"})
    assert r.status_code == 200
    returned = r.json()["words"]
    assert returned and all(w["hsk_level"] == 0 for w in returned)

    eid = r.json()["episode_id"]
    stored = client.get(f"/episodes/{eid}/words").json()
    assert {w["hsk_level"] for w in stored} > {0}  # HSK words were stored too
    assert [w["hsk_level"] for w in stored][-1] == 0  # unknown words sort last


def test_transcript_has_tokens_and_lexicon_for_every_word(client):
    eid = analyze(client)
    data = client.get(f"/episodes/{eid}/transcript").json()
    assert data["episode"]["title"] == "测试 Episode: one"
    assert [s["text"] for s in data["segments"]] == [s.text for s in SEGMENTS]
    for seg in data["segments"]:
        assert "".join(seg["tokens"]) == seg["text"]
    lex = data["lexicon"]
    assert lex["播客"]["definition"] == "podcast (loanword)"
    assert "的" in lex  # function words are clickable even though not extracted as vocabulary


def test_reanalyzing_the_same_url_replaces_instead_of_duplicating(client):
    first = analyze(client)
    second = analyze(client)
    assert first == second
    assert len(client.get("/episodes").json()) == 1
    con = sqlite3.connect(app_module.DB_PATH)
    assert con.execute("select count(*) from segments").fetchone()[0] == len(SEGMENTS)
    words = client.get(f"/episodes/{first}/words").json()
    assert len({w["word"] for w in words}) == len(words)  # no stale duplicates


def test_stream_emits_stages_then_done(client):
    r = client.get("/analyze/stream", params={"url": URL})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    stages = [e["stage"] for e in events]
    assert [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]] == [
        "downloading",
        "transcribing",
        "extracting",
        "done",
    ]
    eid = events[-1]["result"]["episode_id"]
    assert client.get(f"/episodes/{eid}/transcript").json()["segments"]


def test_stream_reports_download_errors(client, monkeypatch):
    def boom(url, out, on_progress=None):
        raise RuntimeError("no network")

    monkeypatch.setattr(app_module, "download_audio", boom)
    r = client.get("/analyze/stream", params={"url": URL})
    last = json.loads(r.text.strip().splitlines()[-1][6:])
    assert last["stage"] == "error" and "Download failed: no network" in last["message"]


def test_transcript_downloads(client):
    eid = analyze(client)
    vtt = client.get(f"/episodes/{eid}/transcript.vtt")
    assert vtt.text.startswith("WEBVTT") and "00:00:03.500 --> 00:00:06.000" in vtt.text
    assert "filename*=UTF-8''%E6%B5%8B%E8%AF%95" in vtt.headers["content-disposition"]
    txt = client.get(f"/episodes/{eid}/transcript.txt").text
    assert txt.splitlines()[0] == "[00:00] 我喜欢听中文播客，学习新的词汇。"


def test_csv_export_respects_level_filter(client):
    eid = analyze(client)
    rows = client.get(f"/episodes/{eid}/export.csv", params={"hsk_levels": "0"}).text.splitlines()
    assert rows[0] == "word,pinyin,definition,hsk_level,frequency,example_sentence"
    assert len(rows) > 1 and all(r.split(",")[3] == "0" for r in rows[1:])


def _apkg_note_count(content: bytes) -> int:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        db = z.read("collection.anki2")
    with tempfile.NamedTemporaryFile(delete=False) as f:  # sqlite needs a real file
        f.write(db)
    try:
        return sqlite3.connect(f.name).execute("select count(*) from notes").fetchone()[0]
    finally:
        os.unlink(f.name)


def test_anki_export_filters_by_level(client):
    eid = analyze(client)
    all_words = client.get(f"/episodes/{eid}/words").json()
    unknown = [w for w in all_words if w["hsk_level"] == 0]

    r = client.get(f"/episodes/{eid}/anki.apkg")
    assert r.status_code == 200 and ".apkg" in r.headers["content-disposition"]
    assert _apkg_note_count(r.content) == len(all_words)

    r0 = client.get(f"/episodes/{eid}/anki.apkg", params={"hsk_levels": "0"})
    assert _apkg_note_count(r0.content) == len(unknown)

    empty_level = next(lv for lv in range(1, 7) if lv not in {w["hsk_level"] for w in all_words})
    r_none = client.get(f"/episodes/{eid}/anki.apkg", params={"hsk_levels": str(empty_level)})
    assert r_none.status_code == 404


def test_unknown_episode_is_404(client):
    for path in ("words", "transcript", "transcript.vtt", "export.csv", "anki.apkg"):
        assert client.get(f"/episodes/999/{path}").status_code == 404


def test_old_database_without_transcripts_still_works(client, tmp_path):
    con = sqlite3.connect(app_module.DB_PATH)
    con.executescript(
        """CREATE TABLE episodes(id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL,
             title TEXT NOT NULL, created_at TEXT NOT NULL);
           CREATE TABLE words(id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
             word TEXT NOT NULL, pinyin TEXT NOT NULL, hsk_level INTEGER NOT NULL,
             frequency INTEGER NOT NULL DEFAULT 1, contexts TEXT NOT NULL DEFAULT '[]');
           INSERT INTO episodes VALUES (1, 'old', 'Old', '2026-01-01');
           INSERT INTO words VALUES (1, 1, '你好', 'nǐ hǎo', 1, 1, '["你好吗"]');"""
    )
    con.commit()
    con.close()
    data = client.get("/episodes/1/transcript").json()
    assert data["segments"] == [] and data["lexicon"] == {}
    assert client.get("/episodes/1/words").json()[0]["word"] == "你好"


def _events(client, **params):
    r = client.get("/analyze/stream", params={"url": URL, **params})
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_stream_reports_download_and_transcription_progress(client, monkeypatch):
    def download(url, out, on_progress=None):
        on_progress(0.5)
        on_progress(1.0)
        return f"{out}/x.mp3"

    def transcribe(path, model_size="base", on_progress=None):
        on_progress(15.0, 30.0)
        on_progress(30.0, 30.0)
        return SEGMENTS

    monkeypatch.setattr(app_module, "download_audio", download)
    monkeypatch.setattr(app_module, "transcribe", transcribe)
    events = _events(client)

    downloading = [e for e in events if e["stage"] == "downloading" and e.get("progress") is not None]
    assert [(e["progress"], e["message"]) for e in downloading] == [
        (0.5, "Downloading audio… 50%"),
        (1.0, "Converting audio to mp3…"),
    ]
    transcribing = [e for e in events if e["stage"] == "transcribing" and e.get("progress") is not None]
    assert transcribing[0]["message"] == "Transcribing… 50% (0:15 of 0:30)"
    assert transcribing[-1]["progress"] == 1.0
    assert all("elapsed" in e for e in events if e["stage"] in ("downloading", "transcribing"))


def test_slow_steps_send_heartbeats_so_the_page_can_show_elapsed_time(client, monkeypatch):
    import time

    def slow_transcribe(path, model_size="base", on_progress=None):
        time.sleep(0.4)  # e.g. a model download that reports nothing
        return SEGMENTS

    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(app_module, "transcribe", slow_transcribe)
    events = _events(client)

    waiting = [e for e in events if e["stage"] == "transcribing"]
    assert len(waiting) >= 3
    assert {e["message"] for e in waiting} == {"Loading base"}
    assert events[-1]["stage"] == "done"


def test_progress_reports_are_throttled():
    sent = []
    report = app_module._throttled(lambda fraction, text: sent.append(fraction), step=0.1)
    for f in (0.0, 0.02, 0.05, 0.11, 0.15, 0.25, 0.99, 1.0):
        report(f, "x")
    assert sent == [0.0, 0.11, 0.25, 0.99, 1.0]
