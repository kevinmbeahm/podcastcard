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
from src.audio import PlaylistLinkError  # noqa: F401  (also reachable as app_module.PlaylistLinkError)
from src.extract import Segment

URL = "https://example.com/ep1"
AUDIO = bytes(range(256)) * 40  # 10,240 bytes of recognisable fake audio


def fake_download(url, out, on_progress=None, **_):
    path = os.path.join(out, "x.mp3")
    with open(path, "wb") as fh:
        fh.write(AUDIO)
    return path
SEGMENTS = [
    Segment(0.0, 3.5, "我喜欢听中文播客，学习新的词汇。"),
    Segment(3.5, 6.0, "播客很有意思。"),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(app_module, "download_audio", fake_download)
    monkeypatch.setattr(
        app_module, "transcribe", lambda path, model_size="base", **_: SEGMENTS
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
    def boom(url, out, on_progress=None, **_):
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
    for path in ("words", "transcript", "audio", "transcript.vtt", "export.csv", "anki.apkg"):
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
    def download(url, out, on_progress=None, **_):
        on_progress(0.5)
        on_progress(1.0)
        return fake_download(url, out)

    def transcribe(path, model_size="base", on_progress=None, **_):
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

    def slow_transcribe(path, model_size="base", **_):
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


# ---------------------------------------------------------------- stored audio


def test_audio_is_kept_and_served(client):
    eid = analyze(client)
    assert client.get(f"/episodes/{eid}/transcript").json()["episode"]["has_audio"] is True

    r = client.get(f"/episodes/{eid}/audio")
    assert r.status_code == 200
    assert r.content == AUDIO
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.headers["accept-ranges"] == "bytes"
    assert "attachment" not in r.headers.get("content-disposition", "")


def test_audio_supports_range_requests_so_the_player_can_seek(client):
    eid = analyze(client)
    r = client.get(f"/episodes/{eid}/audio", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r.content == AUDIO[100:200]
    assert r.headers["content-range"] == f"bytes 100-199/{len(AUDIO)}"


def test_audio_can_be_downloaded_with_the_episode_title(client):
    eid = analyze(client)
    r = client.get(f"/episodes/{eid}/audio", params={"download": "true"})
    disposition = r.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert f'filename="episode_{eid}.mp3"' in disposition
    assert "filename*=UTF-8''%E6%B5%8B%E8%AF%95" in disposition  # the Chinese title


def test_audio_lives_in_a_folder_next_to_the_database(client, tmp_path):
    analyze(client)
    files = list((tmp_path / "podcastcard_audio").iterdir())
    assert len(files) == 1 and files[0].read_bytes() == AUDIO


def test_reanalysing_replaces_the_audio_instead_of_piling_up_copies(client, monkeypatch, tmp_path):
    eid = analyze(client)

    def newer(url, out, on_progress=None, **_):
        path = os.path.join(out, "again.mp3")
        with open(path, "wb") as fh:
            fh.write(b"NEWER-AUDIO")
        return path

    monkeypatch.setattr(app_module, "download_audio", newer)
    assert analyze(client) == eid
    assert client.get(f"/episodes/{eid}/audio").content == b"NEWER-AUDIO"
    assert len(list((tmp_path / "podcastcard_audio").iterdir())) == 1


def test_analysis_still_succeeds_if_the_audio_cannot_be_kept(client, monkeypatch):
    def broken(url, source):
        raise OSError("disk full")

    monkeypatch.setattr(app_module, "_store_audio", broken)
    eid = analyze(client)
    assert client.get(f"/episodes/{eid}/transcript").json()["episode"]["has_audio"] is False
    assert client.get(f"/episodes/{eid}/audio").status_code == 404
    assert client.get(f"/episodes/{eid}/words").json()  # the vocabulary is all there


def test_audio_deleted_from_disk_is_reported_as_missing(client, tmp_path):
    eid = analyze(client)
    for f in (tmp_path / "podcastcard_audio").iterdir():
        f.unlink()
    assert client.get(f"/episodes/{eid}/transcript").json()["episode"]["has_audio"] is False
    assert client.get(f"/episodes/{eid}/audio").status_code == 404


def test_episodes_from_before_audio_was_kept_have_none(client):
    # reuses the old-schema database built in the migration test
    con = sqlite3.connect(app_module.DB_PATH)
    con.executescript(
        """CREATE TABLE episodes(id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL,
             title TEXT NOT NULL, created_at TEXT NOT NULL);
           INSERT INTO episodes VALUES (1, 'old', 'Old', '2026-01-01');"""
    )
    con.commit()
    con.close()
    assert client.get("/episodes/1/transcript").json()["episode"]["has_audio"] is False
    assert client.get("/episodes/1/audio").status_code == 404


def test_the_audio_folder_can_be_configured(client, monkeypatch, tmp_path):
    elsewhere = tmp_path / "my-audio"
    monkeypatch.setenv("PODCASTCARD_AUDIO_DIR", str(elsewhere))
    eid = analyze(client)
    assert [f.read_bytes() for f in elsewhere.iterdir()] == [AUDIO]
    assert client.get(f"/episodes/{eid}/audio").content == AUDIO


# ---------------------------------------------------------------- dictionary


def test_lookup_returns_the_full_entry(client):
    r = client.get("/lookup", params={"word": "行"})
    assert r.status_code == 200
    entry = r.json()
    assert {x["pinyin"] for x in entry["readings"]} >= {"xíng", "háng"}

    traditional = client.get("/lookup", params={"word": "詞彙"}).json()
    assert traditional["simplified"] == "词汇" and traditional["definition"].startswith("vocabulary")


def test_lookup_validates_its_input(client):
    assert client.get("/lookup").status_code == 422
    assert client.get("/lookup", params={"word": ""}).status_code == 422
    assert client.get("/lookup", params={"word": "字" * 41}).status_code == 422
    assert client.get("/lookup", params={"word": "xyzzy"}).json()["readings"] == []


def _stale_episode():
    """An episode saved under an older dictionary: Traditional words nothing could look up."""
    con = sqlite3.connect(app_module.DB_PATH)
    con.executescript(
        """CREATE TABLE episodes(id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL,
             title TEXT NOT NULL, created_at TEXT NOT NULL, lexicon TEXT NOT NULL DEFAULT '{}');
           CREATE TABLE words(id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
             word TEXT NOT NULL, pinyin TEXT NOT NULL, definition TEXT NOT NULL DEFAULT '',
             hsk_level INTEGER NOT NULL, frequency INTEGER NOT NULL DEFAULT 1,
             contexts TEXT NOT NULL DEFAULT '[]');
           CREATE TABLE segments(id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
             idx INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
             tokens TEXT NOT NULL DEFAULT '[]');
           INSERT INTO episodes (id, url, title, created_at, lexicon)
             VALUES (1, 'old', 'Old', '2026-01-01',
                     '{"詞彙": {"pinyin": "", "hsk_level": 0, "definition": ""}}');
           INSERT INTO words VALUES (1, 1, '詞彙', '', '', 0, 1, '["詞彙很重要"]');
           INSERT INTO words VALUES (2, 1, '小孩儿', '', '', 0, 1, '[]');"""
    )
    con.commit()
    con.close()


def test_old_episodes_pick_up_dictionary_improvements_when_opened(client):
    _stale_episode()

    words = {w["word"]: w for w in client.get("/episodes/1/words").json()}
    assert words["詞彙"]["hsk_level"] == 6  # a Traditional word now resolves to its HSK level
    assert words["詞彙"]["definition"].startswith("vocabulary")
    assert words["詞彙"]["pinyin"] == "cí huì"
    assert words["小孩儿"]["definition"] == "erhua variant of 小孩: child"  # was empty

    lexicon = client.get("/episodes/1/transcript").json()["lexicon"]
    assert lexicon["詞彙"]["hsk_level"] == 6 and lexicon["詞彙"]["definition"].startswith("vocabulary")
    assert client.get("/episodes/1/export.csv").text.count("vocabulary") >= 1


def test_the_refresh_happens_once_per_dictionary_version(client, monkeypatch):
    _stale_episode()
    client.get("/episodes/1/words")  # first open: refreshed

    def boom(word):
        raise AssertionError("looked a word up again")

    monkeypatch.setattr(app_module, "get_definition", boom)
    assert client.get("/episodes/1/words").status_code == 200  # no recomputation

    stored = sqlite3.connect(app_module.DB_PATH).execute("select dict_version from episodes").fetchone()[0]
    assert stored == app_module.DICT_VERSION


def test_new_episodes_are_saved_at_the_current_dictionary_version(client):
    eid = analyze(client)
    stored = sqlite3.connect(app_module.DB_PATH).execute(
        "select dict_version from episodes where id = ?", (eid,)
    ).fetchone()[0]
    assert stored == app_module.DICT_VERSION


def test_old_episodes_with_a_transcript_are_re_segmented_when_opened(client, monkeypatch):
    """Traditional text that an older version chopped into single characters."""
    monkeypatch.setattr(app_module, "transcribe", lambda *a, **k: [Segment(0, 3, "我們學習詞彙。")])
    eid = analyze(client)

    con = sqlite3.connect(app_module.DB_PATH)
    con.execute("UPDATE segments SET tokens = ?", (json.dumps(list("我們學習詞彙。"), ensure_ascii=False),))
    con.execute("DELETE FROM words")
    con.execute("UPDATE episodes SET dict_version = 0, lexicon = '{}'")
    con.commit()
    con.close()

    data = client.get(f"/episodes/{eid}/transcript").json()
    assert data["segments"][0]["tokens"] == ["我們", "學習", "詞彙", "。"]
    assert data["lexicon"]["詞彙"]["hsk_level"] == 6
    words = {w["word"]: w for w in client.get(f"/episodes/{eid}/words").json()}
    assert words["詞彙"]["hsk_level"] == 6 and words["學習"]["hsk_level"] == 1
    assert words["詞彙"]["contexts"] == ["我們學習詞彙。"]


def test_episodes_stored_with_the_bare_character_levels_are_corrected_when_opened(client, monkeypatch):
    monkeypatch.setattr(app_module, "transcribe", lambda *a, **k: [Segment(0, 3, "我们拉入这个问题")])
    eid = analyze(client)
    con = sqlite3.connect(app_module.DB_PATH)
    con.execute("UPDATE words SET hsk_level = 6 WHERE word = '入'")  # what the previous list produced
    con.execute("UPDATE episodes SET dict_version = 2")
    con.commit()
    con.close()

    words = {w["word"]: w for w in client.get(f"/episodes/{eid}/words", params={"hsk_levels": "4,5,6"}).json()}
    assert "入" not in words  # no longer a card under the HSK 4-6 filter
    lexicon = client.get(f"/episodes/{eid}/transcript").json()["lexicon"]
    assert lexicon["入"]["hsk_level"] == 0


# ---------------------------------------------------------------- batches (the job queue)

LINKS = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]


def test_a_pasted_list_of_links_is_queued_and_processed_in_the_background(client):
    r = client.post("/jobs", json={"text": "\n".join(LINKS[:2]) + "\n# a note\n" + LINKS[2]})
    assert r.status_code == 200
    assert [j["url"] for j in r.json()["jobs"]] == LINKS
    assert {j["status"] for j in r.json()["jobs"]} == {"queued"}

    assert app_module.jobs.run_pending() == 3  # what the background worker does on its own
    listed = client.get("/jobs").json()["jobs"]
    assert [j["status"] for j in listed] == ["done"] * 3
    assert all(j["episode_id"] and j["title"] == "测试 Episode: one" for j in listed)
    assert len(client.get("/episodes").json()) == 3
    assert client.get(f"/episodes/{listed[0]['episode_id']}/transcript").json()["segments"]


def test_links_can_also_be_sent_as_a_list_and_the_model_is_remembered(client):
    r = client.post("/jobs", json={"urls": LINKS[:1], "model": "small"})
    assert r.json()["jobs"][0]["model"] == "small"


def test_bad_lines_are_reported_but_do_not_block_the_good_ones(client):
    r = client.post("/jobs", json={"text": f"ftp://example.com/x\nnot a link\n{LINKS[0]}\n{LINKS[0]}"})
    body = r.json()
    assert [j["url"] for j in body["jobs"]] == [LINKS[0]]  # the duplicate collapsed
    assert [x["line"] for x in body["rejected"]] == ["ftp://example.com/x", "not a link"]


def test_nothing_to_queue_is_an_error_but_only_bad_links_are_just_reported(client):
    assert client.post("/jobs", json={"text": "  \n# nothing here"}).status_code == 422
    r = client.post("/jobs", json={"text": "ftp://example.com/x"})
    assert r.status_code == 200 and r.json()["jobs"] == [] and r.json()["rejected"]


def test_there_is_a_limit_on_how_many_links_can_be_queued_at_once(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BATCH", 2)
    assert client.post("/jobs", json={"text": "\n".join(LINKS)}).status_code == 422


def test_already_analyzed_videos_are_skipped_unless_asked_otherwise(client):
    eid = analyze(client)  # URL is https://example.com/ep1
    skipped = client.post("/jobs", json={"text": URL}).json()["jobs"][0]
    assert (skipped["status"], skipped["episode_id"]) == ("skipped", eid)
    again = client.post("/jobs", json={"text": URL, "skip_existing": False}).json()["jobs"][0]
    assert again["status"] == "queued"


def test_one_failing_video_is_reported_and_can_be_retried(client, monkeypatch):
    real = app_module.download_audio

    def flaky(url, out, on_progress=None, **kw):
        if url == LINKS[1]:
            raise RuntimeError("video unavailable")
        return real(url, out, on_progress, **kw)

    monkeypatch.setattr(app_module, "download_audio", flaky)
    client.post("/jobs", json={"text": "\n".join(LINKS)})
    app_module.jobs.run_pending()
    jobs = client.get("/jobs").json()["jobs"]
    assert [j["status"] for j in jobs] == ["done", "error", "done"]
    assert jobs[1]["message"] == "Download failed: video unavailable"

    monkeypatch.setattr(app_module, "download_audio", real)
    assert client.post(f"/jobs/{jobs[1]['id']}/retry").json()["status"] == "queued"
    app_module.jobs.run_pending()
    assert client.get("/jobs").json()["jobs"][1]["status"] == "done"


def test_a_playlist_link_fails_with_a_helpful_message(client, monkeypatch):
    def playlist(url):
        raise app_module.PlaylistLinkError("This link is a playlist or channel, not a single video.")

    monkeypatch.setattr(app_module, "_fetch_video_title", playlist)
    client.post("/jobs", json={"text": LINKS[0]})
    app_module.jobs.run_pending()
    job = client.get("/jobs").json()["jobs"][0]
    assert job["status"] == "error" and "playlist or channel" in job["message"]


def test_cancel_retry_and_clear_endpoints(client):
    ids = [j["id"] for j in client.post("/jobs", json={"text": "\n".join(LINKS[:2])}).json()["jobs"]]
    assert client.post(f"/jobs/{ids[0]}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/jobs/{ids[0]}/retry").json()["status"] == "queued"
    assert client.post(f"/jobs/{ids[1]}/retry").status_code == 409  # still waiting
    assert client.post("/jobs/999/cancel").status_code == 404
    assert client.post("/jobs/999/retry").status_code == 404

    app_module.jobs.run_pending()
    assert client.post("/jobs/clear").json() == {"cleared": 2}
    assert client.get("/jobs").json() == {"jobs": []}


def test_the_job_title_is_known_while_the_video_is_still_being_transcribed(client):
    client.post("/jobs", json={"text": LINKS[0]})
    seen = []
    real = app_module._run_pipeline

    def watching(url, model):
        for event in real(url, model):
            yield event
            seen.append(client.get("/jobs").json()["jobs"][0])

    app_module._run_pipeline = watching
    try:
        app_module.jobs.run_pending()
    finally:
        app_module._run_pipeline = real
    live = [j for j in seen if j["stage"] == "transcribing" and j["status"] == "running"]
    assert live and live[-1]["title"] == "测试 Episode: one"


def test_only_one_analysis_runs_at_a_time(monkeypatch):
    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.05)

    def endless(url, model):
        while True:
            yield {"stage": "transcribing", "message": f"working on {url}"}

    monkeypatch.setattr(app_module, "_run_pipeline", endless)
    first = app_module._pipeline("first", "base")
    assert next(first)["message"] == "working on first"  # holds the lock now

    second = app_module._pipeline("second", "base")
    assert next(second)["message"] == "Waiting for another analysis to finish…"
    assert next(second)["message"] == "Waiting for another analysis to finish…"

    first.close()  # the first one is cancelled or finished: the lock is released
    assert next(second)["message"] == "working on second"
    second.close()
    assert app_module._analysis_lock.acquire(blocking=False)  # nothing is left holding it
    app_module._analysis_lock.release()


def test_cancelling_a_running_video_really_stops_its_transcription(client, monkeypatch):
    """The whole chain: cancel -> queue -> pipeline closed -> transcription told to stop."""
    import threading
    import time

    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.05)
    stopped = threading.Event()
    calls = []

    def transcribe(path, model_size="base", should_cancel=None, **_):
        calls.append(path)
        if len(calls) > 1:  # the second video transcribes normally
            return [Segment(0, 1, "你好")]
        while not should_cancel():  # the first one runs until it is told to stop
            time.sleep(0.01)
        stopped.set()  # the real one terminates its child process here
        raise RuntimeError("cancelled")

    monkeypatch.setattr(app_module, "transcribe", transcribe)
    first = client.post("/jobs", json={"text": f"{LINKS[0]}\n{LINKS[1]}"}).json()["jobs"][0]
    threading.Timer(0.4, lambda: client.post(f"/jobs/{first['id']}/cancel")).start()

    started = time.monotonic()
    app_module.jobs.run_pending()

    assert stopped.wait(2), "transcription was never told to stop"
    assert time.monotonic() - started < 10
    assert [j["status"] for j in client.get("/jobs").json()["jobs"]] == ["cancelled", "done"]
