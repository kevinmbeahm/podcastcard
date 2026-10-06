"""Whisper wrapper — transcribe an audio file and return timestamped segments."""

from __future__ import annotations

import logging
from typing import Callable

from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

from .extract import Segment

logger = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "cuda")

# Approximate download sizes, so the UI can say what a first run involves.
MODEL_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 480,
    "medium": 1500,
    "large-v1": 3000,
    "large-v2": 3000,
    "large-v3": 3000,
}

# Called as on_progress(seconds_transcribed, total_seconds)
ProgressCallback = Callable[[float, float], None]


def model_is_cached(model_size: str) -> bool:
    """True if the Whisper model is already on disk (no download needed)."""
    try:
        download_model(model_size, local_files_only=True)
        return True
    except Exception:
        return False


def loading_message(model_size: str) -> str:
    """What the app is doing while the Whisper model is prepared (it may be a download)."""
    if model_is_cached(model_size):
        return f"Loading Whisper model '{model_size}'…"
    size = MODEL_SIZE_MB.get(model_size)
    detail = f"about {size} MB; first use only" if size else "first use only"
    return f"Downloading Whisper model '{model_size}' ({detail})…"


_CUDA_HINT = (
    "GPU transcription needs the CUDA 12 runtime (cuBLAS) and cuDNN 9 installed. "
    "Install them, or use device='cpu'."
)


def _is_cuda_library_error(exc: BaseException) -> bool:
    """True for the errors CTranslate2 raises when CUDA libraries are missing."""
    text = str(exc).lower()
    return any(key in text for key in ("cublas", "cudnn", "cuda"))


def _run(
    audio_path: str,
    model_size: str,
    device: str,
    on_progress: ProgressCallback | None = None,
) -> list[Segment]:
    # int8 is the fast, accurate-enough choice on CPU; "default" lets CUDA use float16.
    compute_type = "int8" if device == "cpu" else "default"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(audio_path, language="zh", beam_size=5)
    # raw_segments is a lazy generator: the actual work (and any CUDA error)
    # happens while iterating it, so this is where progress is reported.
    segments: list[Segment] = []
    for seg in raw_segments:
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
        if on_progress and info.duration:
            on_progress(min(seg.end, info.duration), info.duration)
    return segments


def transcribe(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
    on_progress: ProgressCallback | None = None,
) -> list[Segment]:
    """
    Transcribe *audio_path* using faster-whisper and return a list of
    Segment objects with start/end times and transcript text.

    Parameters
    ----------
    audio_path:
        Path to the audio file (mp3, wav, etc.).
    model_size:
        Whisper model size — "tiny", "base", "small", "medium", "large-v2", etc.
    device:
        "auto" uses the GPU when it works and quietly falls back to the CPU when
        the CUDA libraries are missing; "cpu" and "cuda" force a device.
    on_progress:
        Optional ``(seconds_done, total_seconds)`` callback, called after each segment.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")

    try:
        return _run(audio_path, model_size, device, on_progress)
    except RuntimeError as exc:
        if not _is_cuda_library_error(exc):
            raise
        if device == "auto":
            logger.warning(
                "GPU transcription unavailable (%s). Falling back to CPU.", exc
            )
            return _run(audio_path, model_size, "cpu", on_progress)
        raise RuntimeError(f"{exc}\n{_CUDA_HINT}") from exc
