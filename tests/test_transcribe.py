"""Transcription device handling (faster-whisper is stubbed; no model or GPU needed)."""

import types

import pytest

import src.transcribe as tr


def _fake_model_factory(calls, fail_on):
    """WhisperModel stand-in that fails on the devices listed in *fail_on*."""

    class FakeModel:
        def __init__(self, size, device, compute_type, **kwargs):
            calls.append((device, compute_type))
            self.device = device

        def transcribe(self, path, language, beam_size):
            def gen():
                if self.device in fail_on:  # lazy, like the real generator
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                yield types.SimpleNamespace(start=0.0, end=1.0, text=" 你好 ")

            return gen(), types.SimpleNamespace(duration=10.0)

    return FakeModel


@pytest.fixture(autouse=True)
def _model_not_cached(monkeypatch):
    """Keep tests independent of whatever models are on the developer's disk."""
    monkeypatch.setattr(tr, "model_is_cached", lambda size: False)


def test_auto_falls_back_to_cpu_when_cuda_libs_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(tr, "WhisperModel", _fake_model_factory(calls, {"auto"}))
    segments = tr.transcribe("a.mp3", device="auto")
    assert [s.text for s in segments] == ["你好"]
    assert calls == [("auto", "default"), ("cpu", "int8")]


def test_explicit_cuda_does_not_fall_back_and_explains(monkeypatch):
    calls = []
    monkeypatch.setattr(tr, "WhisperModel", _fake_model_factory(calls, {"cuda"}))
    with pytest.raises(RuntimeError, match="CUDA 12"):
        tr.transcribe("a.mp3", device="cuda")
    assert calls == [("cuda", "default")]


def test_unrelated_runtime_errors_are_not_swallowed(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(tr, "WhisperModel", Boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        tr.transcribe("a.mp3", device="auto")


def test_rejects_unknown_device():
    with pytest.raises(ValueError):
        tr.transcribe("a.mp3", device="tpu")


def test_progress_is_reported_after_each_segment(monkeypatch):
    monkeypatch.setattr(tr, "WhisperModel", _fake_model_factory([], set()))
    seen = []
    tr.transcribe("a.mp3", device="cpu", on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1.0, 10.0)]


def test_model_cache_check_and_loading_messages(monkeypatch, tmp_path):
    complete = tmp_path / "complete"
    complete.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        (complete / name).write_text("x")

    def cached(size, local_files_only=False):
        return str(complete)

    def missing(size, local_files_only=False):
        raise OSError("not in cache")

    monkeypatch.undo()  # drop the autouse stub: exercise the real model_is_cached
    monkeypatch.setattr(tr, "download_model", cached)
    assert tr.model_is_cached("base") is True
    assert tr.loading_message("base") == "Loading Whisper model 'base'…"

    monkeypatch.setattr(tr, "download_model", missing)
    assert tr.model_is_cached("small") is False
    message = tr.loading_message("small")
    assert "Downloading Whisper model 'small'" in message and "480 MB" in message
    assert "first use only" in tr.loading_message("some-custom-model")


def test_interrupted_download_does_not_count_as_cached(monkeypatch, tmp_path):
    """The folder exists and has small files, but model.bin never finished downloading."""
    folder = tmp_path / "snapshot"
    folder.mkdir()
    for name in ("config.json", "tokenizer.json", "vocabulary.txt"):
        (folder / name).write_text("x")

    monkeypatch.undo()
    monkeypatch.setattr(tr, "download_model", lambda size, local_files_only=False: str(folder))
    assert tr.model_is_cached("small") is False
    assert "Downloading Whisper model 'small'" in tr.loading_message("small")


def test_cached_model_loads_offline(monkeypatch):
    seen = {}

    class Model:
        def __init__(self, size, device, compute_type, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(tr, "model_is_cached", lambda size: True)
    monkeypatch.setattr(tr, "WhisperModel", Model)
    tr._load_model("base", "cpu", "int8", tr._no_status)
    assert seen == {"local_files_only": True}


def test_cached_model_that_fails_to_load_is_fetched_again(monkeypatch):
    attempts = []

    class Model:
        def __init__(self, size, device, compute_type, **kwargs):
            attempts.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("corrupt model file")

    monkeypatch.setattr(tr, "model_is_cached", lambda size: True)
    monkeypatch.setattr(tr, "WhisperModel", Model)
    tr._load_model("base", "cpu", "int8", tr._no_status)
    assert attempts == [{"local_files_only": True}, {}]


def test_cuda_errors_while_loading_a_cached_model_are_not_retried_online(monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    monkeypatch.setattr(tr, "model_is_cached", lambda size: True)
    monkeypatch.setattr(tr, "WhisperModel", Model)
    with pytest.raises(RuntimeError, match="cublas"):
        tr._load_model("base", "cuda", "default", tr._no_status)


def test_download_progress_is_measured_from_the_cache_folder(monkeypatch, tmp_path):
    blobs = tmp_path / "models--Systran--faster-whisper-small" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"x" * 1_000_000)
    (blobs / "bbb.incomplete").write_bytes(b"x" * 239_000_000)  # the big file, half done
    monkeypatch.setattr(tr, "_hf_cache_dir", lambda: tmp_path)
    assert tr._downloaded_bytes("small") == 240_000_000
    assert tr._downloaded_bytes("never-downloaded") == 0

    import threading

    updates, stop = [], threading.Event()
    watcher = threading.Thread(
        target=tr._watch_download,
        args=("small", lambda text, fraction: updates.append((text, fraction)), stop, 0.01),
    )
    watcher.start()
    threading.Event().wait(0.1)
    stop.set()
    watcher.join(timeout=2)
    text, fraction = updates[0]
    assert "240 of about 480 MB" in text and fraction == pytest.approx(0.5)


def test_status_messages_walk_through_each_step(monkeypatch):
    monkeypatch.setattr(tr, "WhisperModel", _fake_model_factory([], set()))
    statuses = []
    tr.transcribe("a.mp3", device="cpu", on_status=lambda text, fraction=None: statuses.append(text))
    assert statuses == [
        "Model ready. Decoding the audio…",
        "Transcribing… the first lines can take a minute",
    ]
