"""Download helpers (yt-dlp is replaced by a stand-in: no network needed)."""

import pytest
import yt_dlp
from yt_dlp.utils import DownloadCancelled

import src.audio as audio


class FakeYDL:
    """Records the options and answers extract_info with *info* (or calls the progress hook)."""

    info: dict | Exception = {"title": "A video"}
    options: dict = {}
    progress_events: list[dict] = []

    def __init__(self, opts):
        type(self).options = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        for event in self.progress_events:
            for hook in self.options.get("progress_hooks", []):
                hook(event)
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


@pytest.fixture
def ydl(monkeypatch):
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    FakeYDL.info, FakeYDL.options, FakeYDL.progress_events = {"title": "A video"}, {}, []
    return FakeYDL


# ---------------------------------------------------------------- video_title


def test_title_of_a_single_video(ydl):
    assert audio.video_title("https://example.com/v") == "A video"


def test_a_video_inside_a_playlist_link_means_just_that_video(ydl):
    audio.video_title("https://example.com/watch?v=1&list=2")
    assert ydl.options["noplaylist"] is True


def test_playlist_and_channel_links_are_refused_with_a_clear_message(ydl):
    for info in ({"_type": "playlist", "entries": []}, {"entries": [{"id": "x"}], "title": "Channel"}):
        ydl.info = info
        with pytest.raises(audio.PlaylistLinkError, match="individual videos"):
            audio.video_title("https://example.com/playlist?list=1")


def test_playlists_are_listed_flat_so_checking_one_is_quick(ydl):
    audio.video_title("https://example.com/playlist?list=1")
    assert ydl.options["extract_flat"] == "in_playlist"


def test_unknown_sites_and_network_errors_fall_back_to_the_url(ydl):
    ydl.info = RuntimeError("no network")
    assert audio.video_title("https://example.com/v") == "https://example.com/v"
    ydl.info = {"title": ""}
    assert audio.video_title("https://example.com/v") == "https://example.com/v"


# ---------------------------------------------------------------- download_audio


@pytest.fixture
def downloader(ydl, monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "check_ffmpeg", lambda: None)  # no FFmpeg needed to test the options
    (tmp_path / "A video.mp3").write_bytes(b"x")
    return tmp_path


def test_downloads_only_the_one_video(ydl, downloader):
    audio.download_audio("https://example.com/watch?v=1&list=2", str(downloader))
    assert ydl.options["noplaylist"] is True


def test_a_running_download_can_be_cancelled(ydl, downloader):
    ydl.progress_events = [{"status": "downloading", "downloaded_bytes": 5, "total_bytes": 10}]
    with pytest.raises(DownloadCancelled):
        audio.download_audio("https://example.com/v", str(downloader), should_cancel=lambda: True)


def test_downloads_continue_when_nobody_cancels(ydl, downloader):
    seen = []
    ydl.progress_events = [{"status": "downloading", "downloaded_bytes": 5, "total_bytes": 10}]
    path = audio.download_audio(
        "https://example.com/v", str(downloader), on_progress=seen.append, should_cancel=lambda: False
    )
    assert seen == [0.5] and path.endswith("A video.mp3")
