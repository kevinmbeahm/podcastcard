"""Whisper wrapper — transcribe an audio file and return timestamped segments."""

from __future__ import annotations

from faster_whisper import WhisperModel

from .extract import Segment


def transcribe(audio_path: str, model_size: str = "base") -> list[Segment]:
    """
    Transcribe *audio_path* using faster-whisper and return a list of
    Segment objects with start/end times and transcript text.

    Parameters
    ----------
    audio_path:
        Path to the audio file (mp3, wav, etc.).
    model_size:
        Whisper model size — "tiny", "base", "small", "medium", "large-v2", etc.
    """
    model = WhisperModel(model_size, device="auto", compute_type="default")

    raw_segments, _info = model.transcribe(
        audio_path,
        language="zh",
        beam_size=5,
    )

    segments: list[Segment] = []
    for seg in raw_segments:
        segments.append(
            Segment(start=seg.start, end=seg.end, text=seg.text.strip())
        )

    return segments
