"""Whisper wrapper — transcribe an audio file and return timestamped segments."""

from __future__ import annotations

import logging

from faster_whisper import WhisperModel

from .extract import Segment

logger = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "cuda")

_CUDA_HINT = (
    "GPU transcription needs the CUDA 12 runtime (cuBLAS) and cuDNN 9 installed. "
    "Install them, or use device='cpu'."
)


def _is_cuda_library_error(exc: BaseException) -> bool:
    """True for the errors CTranslate2 raises when CUDA libraries are missing."""
    text = str(exc).lower()
    return any(key in text for key in ("cublas", "cudnn", "cuda"))


def _run(audio_path: str, model_size: str, device: str) -> list[Segment]:
    # int8 is the fast, accurate-enough choice on CPU; "default" lets CUDA use float16.
    compute_type = "int8" if device == "cpu" else "default"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, _info = model.transcribe(audio_path, language="zh", beam_size=5)
    # raw_segments is a lazy generator: CUDA errors can surface while iterating it.
    return [
        Segment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in raw_segments
    ]


def transcribe(
    audio_path: str, model_size: str = "base", device: str = "auto"
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
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")

    try:
        return _run(audio_path, model_size, device)
    except RuntimeError as exc:
        if not _is_cuda_library_error(exc):
            raise
        if device == "auto":
            logger.warning(
                "GPU transcription unavailable (%s). Falling back to CPU.", exc
            )
            return _run(audio_path, model_size, "cpu")
        raise RuntimeError(f"{exc}\n{_CUDA_HINT}") from exc
