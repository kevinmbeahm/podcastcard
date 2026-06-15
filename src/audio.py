"""Audio acquisition — download podcast audio from a URL via yt-dlp."""

from __future__ import annotations

import os

import yt_dlp


def download_audio(url: str, output_dir: str) -> str:
    """
    Download audio from *url* to *output_dir* and return the path to the
    resulting mp3 file.

    Uses yt-dlp's Python API to fetch the best available audio stream and
    post-process it to mp3 via ffmpeg.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    downloaded_path: list[str] = []

    class _InfoHook:
        def __init__(self) -> None:
            self.filepath: str | None = None

        def __call__(self, d: dict) -> None:
            if d["status"] == "finished":
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
