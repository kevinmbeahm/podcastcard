"""Whisper wrapper — transcribe an audio file and return timestamped segments."""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import threading
from pathlib import Path
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

# Files a usable faster-whisper model folder must contain.
_REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")

# Called as on_progress(seconds_transcribed, total_seconds)
ProgressCallback = Callable[[float, float], None]
# Called as on_status(text, fraction) where fraction is 0-1 or None if unknown
StatusCallback = Callable[[str, "float | None"], None]


def _no_status(text: str, fraction: float | None = None) -> None:
    pass


def model_is_cached(model_size: str) -> bool:
    """True if the Whisper model is fully on disk (no download needed).

    ``download_model(local_files_only=True)`` only checks that the cache folder
    exists, so an interrupted download (small files present, ``model.bin``
    missing) would look cached. Check the key files too.
    """
    try:
        folder = Path(download_model(model_size, local_files_only=True))
    except Exception:
        return False
    return all((folder / name).is_file() for name in _REQUIRED_MODEL_FILES)


def loading_message(model_size: str) -> str:
    """What the app is doing while the Whisper model is prepared (it may be a download)."""
    if model_is_cached(model_size):
        return f"Loading Whisper model '{model_size}'…"
    size = MODEL_SIZE_MB.get(model_size)
    detail = f"about {size} MB; first use only" if size else "first use only"
    return f"Downloading Whisper model '{model_size}' ({detail})…"


def _hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path(os.path.expanduser("~/.cache/huggingface/hub"))


def _downloaded_bytes(model_size: str) -> int:
    """Bytes of the model on disk so far, including a partly downloaded file."""
    blobs = _hf_cache_dir() / f"models--Systran--faster-whisper-{model_size}" / "blobs"
    try:
        return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
    except OSError:
        return 0


def _watch_download(
    model_size: str, status: StatusCallback, stop: threading.Event, interval: float = 1.0
) -> None:
    """Report model download progress by watching the cache folder grow."""
    expected_mb = MODEL_SIZE_MB.get(model_size)
    if not expected_mb:
        return
    while not stop.wait(interval):
        done_mb = _downloaded_bytes(model_size) / 1e6
        if done_mb > 0:
            status(
                f"Downloading Whisper model '{model_size}'… {done_mb:.0f} of about {expected_mb} MB",
                min(done_mb / expected_mb, 0.99),
            )


_CUDA_HINT = (
    "GPU transcription needs the CUDA 12 runtime (cuBLAS) and cuDNN 9 installed. "
    "Install them, or use device='cpu'."
)


def _is_cuda_library_error(exc: BaseException) -> bool:
    """True for the errors CTranslate2 raises when CUDA libraries are missing."""
    text = str(exc).lower()
    return any(key in text for key in ("cublas", "cudnn", "cuda"))


def _load_model(
    model_size: str, device: str, compute_type: str, status: StatusCallback
) -> WhisperModel:
    # A model that is fully on disk loads offline, so a slow or unreachable
    # Hugging Face can't stall it (the default is to check for updates online).
    if model_is_cached(model_size):
        status(f"Loading Whisper model '{model_size}'…", None)
        try:
            return WhisperModel(
                model_size, device=device, compute_type=compute_type, local_files_only=True
            )
        except Exception as exc:
            if _is_cuda_library_error(exc):
                raise  # not a model problem: let transcribe() fall back to the CPU
            logger.warning("Cached model failed to load (%s); fetching it again.", exc)

    stop = threading.Event()
    threading.Thread(
        target=_watch_download, args=(model_size, status, stop), daemon=True
    ).start()
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    finally:
        stop.set()


def _run(
    audio_path: str,
    model_size: str,
    device: str,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
) -> list[Segment]:
    status = on_status or _no_status
    # int8 is the fast, accurate-enough choice on CPU; "default" lets CUDA use float16.
    compute_type = "int8" if device == "cpu" else "default"
    model = _load_model(model_size, device, compute_type, status)

    status("Model ready. Decoding the audio…", None)
    raw_segments, info = model.transcribe(audio_path, language="zh", beam_size=5)

    status("Transcribing… the first lines can take a minute", None)
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
    on_status: StatusCallback | None = None,
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
    on_status:
        Optional ``(text, fraction)`` callback describing the step before transcription
        starts: model download (with a fraction), model loading, audio decoding.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")

    try:
        return _run(audio_path, model_size, device, on_progress, on_status)
    except RuntimeError as exc:
        if not _is_cuda_library_error(exc):
            raise
        if device == "auto":
            logger.warning(
                "GPU transcription unavailable (%s). Falling back to CPU.", exc
            )
            return _run(audio_path, model_size, "cpu", on_progress, on_status)
        raise RuntimeError(f"{exc}\n{_CUDA_HINT}") from exc


# ---------------------------------------------------------------------------
# Running each transcription in its own process
# ---------------------------------------------------------------------------
# A long-lived server that runs Whisper in-process keeps native state alive between
# jobs (CTranslate2/OpenMP thread pools, GPU memory). A fresh process per job starts
# clean every time, frees all memory when it exits, can be cancelled by killing it,
# and a native crash or hang can't take the web server down with it.


class TranscriptionCancelled(RuntimeError):
    """The caller no longer wants the result (e.g. the browser tab was closed)."""


def _subprocess_main(messages, audio_path: str, model_size: str, device: str) -> None:
    """Entry point of the child process: transcribe and send everything back as messages."""
    try:
        segments = transcribe(
            audio_path,
            model_size,
            device,
            on_progress=lambda done, total: messages.put(("progress", done, total)),
            on_status=lambda text, fraction=None: messages.put(("status", text, fraction)),
        )
        messages.put(("result", [(s.start, s.end, s.text) for s in segments]))
    except BaseException as exc:  # reported to the parent, which re-raises
        messages.put(("error", f"{type(exc).__name__}: {exc}"))


def transcribe_isolated(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    poll_seconds: float = 0.5,
    _target: Callable = _subprocess_main,  # replaceable in tests
) -> list[Segment]:
    """Like :func:`transcribe`, but runs in a fresh child process.

    Progress and status callbacks are relayed from the child. If *should_cancel*
    returns True the child is terminated and :class:`TranscriptionCancelled` is raised.
    Raises RuntimeError if the child fails or dies without a result.
    """
    ctx = multiprocessing.get_context("spawn")  # a clean interpreter, same on every OS
    messages = ctx.Queue()
    process = ctx.Process(
        target=_target, args=(messages, audio_path, model_size, device), daemon=True
    )
    process.start()
    try:
        while True:
            if should_cancel and should_cancel():
                raise TranscriptionCancelled("Transcription was cancelled.")
            try:
                message = messages.get(timeout=poll_seconds)
            except queue.Empty:
                if process.is_alive():
                    continue
                try:  # it may have sent its result just before exiting
                    message = messages.get(timeout=2.0)
                except queue.Empty:
                    raise RuntimeError(
                        "The transcription process stopped unexpectedly "
                        f"(exit code {process.exitcode}). Check the server terminal for details."
                    ) from None

            kind = message[0]
            if kind == "progress" and on_progress:
                on_progress(message[1], message[2])
            elif kind == "status" and on_status:
                on_status(message[1], message[2])
            elif kind == "result":
                return [Segment(start, end, text) for start, end, text in message[1]]
            elif kind == "error":
                raise RuntimeError(message[1])
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        messages.close()
