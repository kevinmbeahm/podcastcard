"""Transcription device handling (faster-whisper is stubbed; no model or GPU needed)."""

import types

import pytest

import src.transcribe as tr


def _fake_model_factory(calls, fail_on):
    """WhisperModel stand-in that fails on the devices listed in *fail_on*."""

    class FakeModel:
        def __init__(self, size, device, compute_type):
            calls.append((device, compute_type))
            self.device = device

        def transcribe(self, path, language, beam_size):
            def gen():
                if self.device in fail_on:  # lazy, like the real generator
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                yield types.SimpleNamespace(start=0.0, end=1.0, text=" 你好 ")

            return gen(), None

    return FakeModel


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
