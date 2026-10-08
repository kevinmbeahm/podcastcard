"""Audio acquisition — download podcast audio from a URL via yt-dlp."""

from __future__ import annotations

import os
import shutil
from typing import Callable

import yt_dlp
from yt_dlp.utils import DownloadCancelled


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg/ffprobe are required to convert downloaded audio to mp3."""


_FFMPEG_HELP = (
    "FFmpeg is required but ffmpeg/ffprobe were not found on your PATH. Install it:\n"
    "  macOS:   brew install ffmpeg\n"
    "  Ubuntu:  sudo apt install ffmpeg\n"
    "  Windows: winget install Gyan.FFmpeg   (then reopen your terminal)"
)


def check_ffmpeg() -> None:
    """Raise FFmpegNotFoundError with install hints if FFmpeg is unavailable."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise FFmpegNotFoundError(_FFMPEG_HELP)


class PlaylistLinkError(ValueError):
    """The link is a playlist or channel, not a single video."""


def download_audio(
    url: str,
    output_dir: str,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    """
    Download audio from *url* to *output_dir* and return the path to the
    resulting mp3 file.

    Uses yt-dlp's Python API to fetch the best available audio stream and
    post-process it to mp3 via ffmpeg.

    *on_progress*, if given, is called with the download fraction (0.0-1.0).
    It reaches 1.0 when the download ends, before the mp3 conversion starts.

    *should_cancel*, if given, is checked while the file downloads; when it returns True the
    download is abandoned and yt-dlp's ``DownloadCancelled`` is raised.
    """
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    downloaded_path: list[str] = []

    class _InfoHook:
        def __init__(self) -> None:
            self.filepath: str | None = None

        def __call__(self, d: dict) -> None:
            if should_cancel and should_cancel():
                raise DownloadCancelled("Download cancelled.")
            if d["status"] == "downloading" and on_progress:
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if total:
                    on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))
            elif d["status"] == "finished":
                if on_progress:
                    on_progress(1.0)
                # after post-processing the file extension changes to mp3
                self.filepath = os.path.splitext(d["filename"])[0] + ".mp3"

    hook = _InfoHook()

    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
        # A video link that also names a playlist (watch?v=…&list=…) means just that video.
        "noplaylist": True,
        # Fail (and retry) instead of waiting forever on a stalled connection.
        "socket_timeout": 30,
        "retries": 5,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if hook.filepath and os.path.exists(hook.filepath):
            return hook.filepath
        # Fallback: reconstruct path from info dict
        title = info.get("title", "audio")
        # sanitise title the same way yt-dlp does (basic)
        safe_title = yt_dlp.utils.sanitize_filename(title)
        fallback = os.path.join(output_dir, f"{safe_title}.mp3")
        if os.path.exists(fallback):
            return fallback
        # Last resort: search output_dir for any mp3 newer than this call
        mp3s = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".mp3")
        ]
        if mp3s:
            return max(mp3s, key=os.path.getmtime)
        raise FileNotFoundError(
            f"yt-dlp finished but could not locate the downloaded mp3 in {output_dir}"
        )


def video_title(url: str) -> str:
    """The title of the video at *url*, without downloading it.

    Falls back to the URL itself if yt-dlp cannot say (offline, unsupported site, ...).
    Raises :class:`PlaylistLinkError` if the link is a playlist or channel: downloading that
    would fetch every video into one file and keep only the last.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": "in_playlist",  # list a playlist's videos without opening each one
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return url
    if not isinstance(info, dict):
        return url
    if info.get("_type") == "playlist" or "entries" in info:
        raise PlaylistLinkError(
            "This link is a playlist or channel, not a single video. "
            "Paste the links of the individual videos instead."
        )
    return info.get("title") or url
