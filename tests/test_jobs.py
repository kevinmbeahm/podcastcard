"""The batch queue: parsing pasted links, ordering, skipping, cancelling, retrying, recovery."""

import sqlite3
import time

import pytest

from src.jobs import SCHEMA, JobQueue, parse_urls

A, B, C = "https://example.com/a", "https://example.com/b", "https://example.com/c"


# ---------------------------------------------------------------- parsing pasted links


def test_one_link_per_line_with_blank_lines_and_comments():
    text = f"{A}\n\n  {B}  \n# a note to myself\n{C}\n"
    assert parse_urls(text) == ([A, B, C], [])


def test_links_separated_by_spaces_work_too_and_duplicates_collapse():
    links, rejected = parse_urls(f"{A} {B}\n{A}")
    assert links == [A, B] and rejected == []


def test_a_list_of_pieces_is_accepted():
    assert parse_urls(["", f"{A}\n{B}", C])[0] == [A, B, C]


@pytest.mark.parametrize("bad", ["not a link", "ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)", "http://"])
def test_only_web_links_are_accepted(bad):
    links, rejected = parse_urls(f"{A}\n{bad}")
    assert links == [A]
    assert [r["line"] for r in rejected] == [bad]  # the whole line, reported once
    assert "http" in rejected[0]["reason"]


def test_words_around_a_link_are_ignored():
    text = f"Episode 3 – {A}\nmy favourite: {B} (really good)"
    assert parse_urls(text) == ([A, B], [])


def test_a_line_with_a_bad_scheme_next_to_a_good_link_still_works():
    assert parse_urls(f"ftp://example.com/x {A}") == ([A], [])


def test_a_url_with_a_fragment_is_not_mistaken_for_a_comment():
    assert parse_urls("https://example.com/a#t=30")[0] == ["https://example.com/a#t=30"]


# ---------------------------------------------------------------- the queue


@pytest.fixture
def connect(tmp_path):
    path = tmp_path / "jobs.db"

    def open_db():
        con = sqlite3.connect(path, timeout=30)
        con.row_factory = sqlite3.Row
        con.executescript(
            "CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, url TEXT UNIQUE, title TEXT);" + SCHEMA
        )
        return con

    return open_db


def finishing(url, model):
    """A pipeline that analyses successfully."""
    yield {"stage": "downloading", "message": "Fetching audio…", "progress": None, "elapsed": 0}
    yield {"stage": "transcribing", "message": "Transcribing… 50%", "progress": 0.5, "elapsed": 4, "title": f"T:{url[-1]}"}
    yield {"stage": "done", "message": "Found 5 words.", "result": {"episode_id": 7, "title": f"T:{url[-1]}"}}


def statuses(queue):
    return [job["status"] for job in queue.list_jobs()]


def test_links_are_queued_in_order(connect):
    queue = JobQueue(connect, finishing)
    result = queue.enqueue([A, B, C], model="small")
    assert [j["url"] for j in result["jobs"]] == [A, B, C]
    assert {j["status"] for j in result["jobs"]} == {"queued"}
    assert {j["model"] for j in result["jobs"]} == {"small"}
    assert result["rejected"] == []


def test_jobs_run_one_after_another_and_record_the_result(connect):
    order = []

    def pipeline(url, model):
        order.append((url, model))
        yield from finishing(url, model)

    queue = JobQueue(connect, pipeline)
    queue.enqueue([A, B, C], model="tiny")
    assert queue.run_pending() == 3
    assert order == [(A, "tiny"), (B, "tiny"), (C, "tiny")]
    assert statuses(queue) == ["done"] * 3
    job = queue.list_jobs()[0]
    assert (job["episode_id"], job["title"], job["progress"], job["stage"]) == (7, "T:a", 1.0, "done")
    assert job["started_at"] and job["finished_at"]


def test_progress_is_stored_while_a_job_runs(connect):
    queue = JobQueue(connect, lambda url, model: finishing(url, model))
    seen = []

    def pipeline(url, model):
        yield {"stage": "downloading", "message": "Fetching audio…", "progress": None, "elapsed": 0}
        yield {"stage": "transcribing", "message": "Transcribing… 42%", "progress": 0.42, "elapsed": 9, "title": "Live title"}
        seen.append(queue.list_jobs()[0])  # resumed after the second event was written
        yield {"stage": "done", "message": "ok", "result": {"episode_id": 1, "title": "Live title"}}

    queue._pipeline = pipeline
    queue.enqueue([A])
    queue.run_pending()
    live = seen[0]
    assert live["status"] == "running"
    assert (live["stage"], live["message"], live["progress"], live["elapsed"], live["title"]) == (
        "transcribing",
        "Transcribing… 42%",
        0.42,
        9,
        "Live title",
    )


def test_one_failing_video_does_not_stop_the_rest(connect):
    def pipeline(url, model):
        if url == B:
            yield {"stage": "error", "message": "Download failed: video unavailable"}
            return
        yield from finishing(url, model)

    queue = JobQueue(connect, pipeline)
    queue.enqueue([A, B, C])
    queue.run_pending()
    jobs = queue.list_jobs()
    assert [j["status"] for j in jobs] == ["done", "error", "done"]
    assert jobs[1]["message"] == "Download failed: video unavailable"


def test_a_pipeline_that_crashes_is_reported_as_an_error(connect):
    def pipeline(url, model):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    queue = JobQueue(connect, pipeline)
    queue.enqueue([A, B])
    assert queue.run_pending() == 2
    assert statuses(queue) == ["error", "error"]
    assert queue.list_jobs()[0]["message"] == "Unexpected error: boom"


def test_a_pipeline_that_ends_without_a_result_is_an_error(connect):
    queue = JobQueue(connect, lambda url, model: iter([{"stage": "downloading", "message": "x"}]))
    queue.enqueue([A])
    queue.run_pending()
    assert queue.list_jobs()[0]["status"] == "error"


# ---------------------------------------------------------------- duplicates and "already analyzed"


def test_a_link_already_waiting_is_not_queued_twice(connect):
    queue = JobQueue(connect, finishing)
    queue.enqueue([A])
    again = queue.enqueue([A, B])
    assert [j["url"] for j in again["jobs"]] == [B]
    assert again["rejected"] == [{"line": A, "reason": "Already in the queue"}]


def test_a_link_can_be_queued_again_once_it_has_finished(connect):
    queue = JobQueue(connect, finishing)
    queue.enqueue([A], skip_existing=False)
    queue.run_pending()
    assert len(queue.enqueue([A], skip_existing=False)["jobs"]) == 1


def test_already_analyzed_videos_are_skipped(connect):
    con = connect()
    con.execute("INSERT INTO episodes (id, url, title) VALUES (5, ?, 'Old episode')", (A,))
    con.commit()
    con.close()

    ran = []
    queue = JobQueue(connect, lambda url, model: (ran.append(url), finishing(url, model))[1])
    jobs = queue.enqueue([A, B])["jobs"]
    assert [j["status"] for j in jobs] == ["skipped", "queued"]
    assert (jobs[0]["episode_id"], jobs[0]["title"], jobs[0]["message"]) == (5, "Old episode", "Already analyzed")

    queue.run_pending()
    assert ran == [B]  # the skipped one was never processed


def test_skip_existing_can_be_switched_off(connect):
    con = connect()
    con.execute("INSERT INTO episodes (id, url, title) VALUES (5, ?, 'Old episode')", (A,))
    con.commit()
    con.close()
    queue = JobQueue(connect, finishing)
    assert queue.enqueue([A], skip_existing=False)["jobs"][0]["status"] == "queued"


# ---------------------------------------------------------------- cancel, retry, clear


def test_a_waiting_job_can_be_cancelled_and_is_never_run(connect):
    ran = []
    queue = JobQueue(connect, lambda url, model: (ran.append(url), finishing(url, model))[1])
    ids = [j["id"] for j in queue.enqueue([A, B])["jobs"]]
    assert queue.cancel(ids[0])["status"] == "cancelled"
    queue.run_pending()
    assert ran == [B] and statuses(queue) == ["cancelled", "done"]


def test_cancelling_a_running_job_closes_its_pipeline_and_the_queue_moves_on(connect):
    closed = []
    queue = JobQueue(connect, finishing)

    def pipeline(url, model):
        if url != A:
            yield from finishing(url, model)
            return
        try:
            for _ in range(1000):
                if _ == 2:
                    queue.cancel(queue.list_jobs()[0]["id"])  # the user clicks cancel mid-run
                yield {"stage": "transcribing", "message": "Transcribing…", "progress": 0.1, "elapsed": 1}
        finally:
            closed.append(url)  # what stops the child process in the real pipeline

    queue._pipeline = pipeline
    queue.enqueue([A, B])
    queue.run_pending()
    assert closed == [A]
    assert statuses(queue) == ["cancelled", "done"]
    assert queue.list_jobs()[0]["message"] == "Cancelled"


def test_cancel_edge_cases(connect):
    queue = JobQueue(connect, finishing)
    assert queue.cancel(999) is None
    job = queue.enqueue([A])["jobs"][0]
    queue.run_pending()
    assert queue.cancel(job["id"])["status"] == "done"  # too late: nothing changes


def test_failed_and_cancelled_jobs_can_be_retried(connect):
    attempts = []

    def flaky(url, model):
        attempts.append(url)
        if len(attempts) == 1:
            yield {"stage": "error", "message": "Download failed: timeout"}
        else:
            yield from finishing(url, model)

    queue = JobQueue(connect, flaky)
    job = queue.enqueue([A])["jobs"][0]
    queue.run_pending()
    assert queue.list_jobs()[0]["status"] == "error"

    retried = queue.retry(job["id"])
    assert (retried["status"], retried["message"], retried["progress"]) == ("queued", "", None)
    queue.run_pending()
    assert queue.list_jobs()[0]["status"] == "done" and attempts == [A, A]


def test_retry_refuses_jobs_that_cannot_be_retried(connect):
    queue = JobQueue(connect, finishing)
    job = queue.enqueue([A])["jobs"][0]
    with pytest.raises(ValueError):
        queue.retry(job["id"])  # still waiting
    queue.run_pending()
    with pytest.raises(ValueError):
        queue.retry(job["id"])  # done
    assert queue.retry(999) is None


def test_clearing_removes_only_finished_jobs(connect):
    queue = JobQueue(connect, finishing)
    done, waiting, cancelled = (j["id"] for j in queue.enqueue([A, B, C])["jobs"])
    queue.cancel(cancelled)
    queue._run_one(queue._claim_next())  # runs the oldest waiting job (A) and leaves B waiting
    assert statuses(queue) == ["done", "queued", "cancelled"]
    assert queue.clear_finished() == 2
    assert [j["id"] for j in queue.list_jobs()] == [waiting]


# ---------------------------------------------------------------- restarts and the real worker thread


def test_jobs_interrupted_by_a_restart_are_queued_again(connect):
    queue = JobQueue(connect, finishing)
    queue.enqueue([A, B])
    con = connect()
    con.execute("UPDATE jobs SET status = 'running', stage = 'transcribing' WHERE url = ?", (A,))
    con.commit()
    con.close()
    assert statuses(queue) == ["running", "queued"]  # as a crashed server would have left it

    queue.start()
    queue.stop(wait=True)  # let the worker finish whatever it had begun
    states = statuses(queue)
    assert "running" not in states  # never left "running" forever
    assert states[0] in ("queued", "done")  # picked up again (or about to be)


def test_the_background_worker_processes_the_queue_by_itself(connect):
    queue = JobQueue(connect, finishing)
    queue.start()
    try:
        queue.enqueue([A, B, C])
        deadline = time.time() + 15
        while time.time() < deadline and statuses(queue) != ["done"] * 3:
            time.sleep(0.05)
        assert statuses(queue) == ["done"] * 3
        queue.enqueue([A], skip_existing=False)  # and it keeps listening afterwards
        deadline = time.time() + 15
        while time.time() < deadline and statuses(queue)[-1] != "done":
            time.sleep(0.05)
        assert statuses(queue) == ["done"] * 4
    finally:
        queue.stop(wait=True)
