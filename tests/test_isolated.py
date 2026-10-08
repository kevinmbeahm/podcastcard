"""Transcription in a child process (real processes, stand-in work: no model needed)."""

import os
import threading
import time

import pytest

import _isolation_targets as targets
import src.app as app_module
from src.transcribe import TranscriptionCancelled, transcribe_isolated


def run(target, **kwargs):
    return transcribe_isolated("a.mp3", "base", "cpu", _target=target, **kwargs)


def test_result_and_callbacks_come_back_from_the_child():
    statuses, progress = [], []
    segments = run(
        targets.succeeds,
        on_status=lambda text, fraction=None: statuses.append((text, fraction)),
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert [(s.start, s.end, s.text) for s in segments] == [(0.0, 1.0, "a.mp3|base|cpu")]
    assert statuses == [("Model ready", None)]
    assert progress == [(5.0, 10.0)]


def test_every_call_gets_a_fresh_process():
    first = run(targets.reports_pid)[0].text
    second = run(targets.reports_pid)[0].text
    assert first != second
    assert str(os.getpid()) not in (first, second)  # and neither is the server process


def test_errors_in_the_child_are_raised_in_the_parent():
    with pytest.raises(RuntimeError, match="ValueError: bad audio"):
        run(targets.fails)


def test_a_child_that_dies_without_a_result_is_reported():
    with pytest.raises(RuntimeError, match=r"stopped unexpectedly \(exit code 3\)"):
        run(targets.crashes)


def test_cancelling_terminates_the_child_promptly():
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(TranscriptionCancelled):
        run(targets.runs_forever, should_cancel=cancel.is_set, poll_seconds=0.1)
    assert time.monotonic() - started < 20  # the child sleeps for 60 s


def test_closing_the_progress_generator_cancels_the_worker():
    """What happens when the browser tab is closed mid-analysis."""
    saw_cancel = threading.Event()

    def work(report):
        while not report.cancelled.is_set():
            time.sleep(0.01)
        saw_cancel.set()

    events = app_module._run_with_progress(work, "transcribing", "Working", time.monotonic())
    next(events)  # starts the worker
    events.close()
    assert saw_cancel.wait(timeout=3)
