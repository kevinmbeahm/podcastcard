"""CLI wiring tests (pipeline stages are stubbed; no network or audio needed)."""

import sys
import types

from typer.testing import CliRunner

from src.cli import app
from src.extract import Segment

runner = CliRunner()


def _stub_pipeline(monkeypatch):
    audio = types.ModuleType("src.audio")
    audio.download_audio = lambda url, out, on_progress=None: f"{out}/fake.mp3"
    audio.FFmpegNotFoundError = type("FFmpegNotFoundError", (RuntimeError,), {})
    transcribe = types.ModuleType("src.transcribe")
    transcribe.transcribe = lambda path, model_size="base", device="auto", on_progress=None, on_status=None: [
        Segment(0, 3, "我喜欢听中文播客，学习新的词汇。")
    ]
    transcribe.loading_message = lambda model: f"Loading Whisper model '{model}'…"
    monkeypatch.setitem(sys.modules, "src.audio", audio)
    monkeypatch.setitem(sys.modules, "src.transcribe", transcribe)


def test_run_subcommand_accepts_url(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch)
    url = "https://www.youtube.com/watch?v=aYgDExQVn_I"
    result = runner.invoke(app, ["run", url, "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "播客" in result.output
    csv_text = (tmp_path / "words.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "word,pinyin,definition,hsk_level,frequency,contexts"
    assert "podcast" in csv_text


def test_hsk_filter_limits_output(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch)
    result = runner.invoke(
        app, ["run", "http://x", "--hsk-levels", "0", "--output", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    rows = (tmp_path / "words.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert rows and all(r.split(",")[3] == "0" for r in rows)


def test_missing_ffmpeg_gives_install_hint(monkeypatch, tmp_path):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = runner.invoke(app, ["run", "http://x", "--output", str(tmp_path)])
    assert result.exit_code == 1
    assert "brew install ffmpeg" in result.output
    assert "Traceback" not in result.output


def test_run_writes_transcript_files_and_anki_deck(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch)
    result = runner.invoke(app, ["run", "http://x", "--output", str(tmp_path), "--anki"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8").startswith("[00:00] 我喜欢")
    assert (tmp_path / "transcript.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert (tmp_path / "anki_deck.apkg").stat().st_size > 0


def test_anki_deck_is_opt_in(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch)
    runner.invoke(app, ["run", "http://x", "--output", str(tmp_path)])
    assert not (tmp_path / "anki_deck.apkg").exists()


def test_serve_starts_uvicorn_with_the_app(monkeypatch):
    calls = []
    fake = types.ModuleType("uvicorn")
    fake.run = lambda target, **kwargs: calls.append((target, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    result = runner.invoke(app, ["serve", "--port", "9001"])
    assert result.exit_code == 0, result.output
    assert calls == [("src.app:app", {"host": "127.0.0.1", "port": 9001, "reload": False})]
    assert "http://localhost:9001" in result.output


# ---------------------------------------------------------------- several videos at once


def _stub_pipeline_named(monkeypatch, fail_on=(), calls=None):
    """Like _stub_pipeline, but the audio file is named after the URL's last part."""
    audio = types.ModuleType("src.audio")
    audio.FFmpegNotFoundError = type("FFmpegNotFoundError", (RuntimeError,), {})

    def download_audio(url, out, on_progress=None):
        if calls is not None:
            calls.append(url)
        if url in fail_on:
            raise RuntimeError("video unavailable")
        return f"{out}/{url.rsplit('/', 1)[-1]}.mp3"

    audio.download_audio = download_audio
    transcribe = types.ModuleType("src.transcribe")
    transcribe.transcribe = lambda path, model_size="base", device="auto", on_progress=None, on_status=None: [
        Segment(0, 3, "我喜欢听中文播客，学习新的词汇。")
    ]
    transcribe.loading_message = lambda model: "Loading"
    monkeypatch.setitem(sys.modules, "src.audio", audio)
    monkeypatch.setitem(sys.modules, "src.transcribe", transcribe)
    return audio


U1, U2, U3 = "https://example.com/ep-one", "https://example.com/ep-two", "https://example.com/ep-three"


def test_several_videos_each_get_their_own_folder(monkeypatch, tmp_path):
    _stub_pipeline_named(monkeypatch)
    result = runner.invoke(app, ["run", U1, U2, "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["01-ep-one", "02-ep-two"]  # no staging folders left
    for folder in ("01-ep-one", "02-ep-two"):
        assert (tmp_path / folder / "words.csv").exists()
        assert (tmp_path / folder / "transcript.txt").read_text(encoding="utf-8").startswith("[00:00]")
    assert "Batch summary" in result.output and "2 of 2 videos done" in result.output


def test_a_file_of_urls_works_with_comments_blank_lines_and_duplicates(monkeypatch, tmp_path):
    calls = []
    _stub_pipeline_named(monkeypatch, calls=calls)
    listing = tmp_path / "urls.txt"
    listing.write_text(f"# my podcasts\n{U1}\n\n  {U2}  \n{U1}\n", encoding="utf-8")
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", U3, "--file", str(listing), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert calls == [U3, U1, U2]  # positional first, then the file; the duplicate is processed once
    assert sorted(p.name for p in out.iterdir()) == ["01-ep-three", "02-ep-one", "03-ep-two"]


def test_one_failing_video_does_not_stop_the_others_and_sets_the_exit_code(monkeypatch, tmp_path):
    _stub_pipeline_named(monkeypatch, fail_on=(U2,))
    result = runner.invoke(app, ["run", U1, U2, U3, "--output", str(tmp_path)])
    assert result.exit_code == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["01-ep-one", "03-ep-three"]  # nothing left of #2
    assert "video unavailable" in result.output and "2 of 3 videos done" in result.output
    assert "1 failed" in result.output


def test_missing_ffmpeg_stops_the_whole_batch_after_one_attempt(monkeypatch, tmp_path):
    calls = []
    audio = _stub_pipeline_named(monkeypatch, calls=calls)

    def no_ffmpeg(url, out, on_progress=None):
        calls.append(url)
        raise audio.FFmpegNotFoundError("FFmpeg is required but was not found")

    audio.download_audio = no_ffmpeg
    result = runner.invoke(app, ["run", U1, U2, "--output", str(tmp_path)])
    assert result.exit_code == 1 and calls == [U1]
    assert "FFmpeg is required" in result.output
    assert list(tmp_path.iterdir()) == []  # no leftover working folder


def test_anki_decks_are_written_per_video(monkeypatch, tmp_path):
    _stub_pipeline_named(monkeypatch)
    result = runner.invoke(app, ["run", U1, U2, "--anki", "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "01-ep-one" / "anki_deck.apkg").stat().st_size > 0
    assert (tmp_path / "02-ep-two" / "anki_deck.apkg").stat().st_size > 0


def test_chinese_titles_become_readable_folder_names(monkeypatch, tmp_path):
    audio = _stub_pipeline_named(monkeypatch)
    audio.download_audio = lambda url, out, on_progress=None: f"{out}/第3集：你好？世界.mp3"
    result = runner.invoke(app, ["run", U1, U2, "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    # the same title twice still gives two folders, told apart by their number
    assert sorted(p.name for p in tmp_path.iterdir()) == ["01-第3集-你好-世界", "02-第3集-你好-世界"]


def test_folder_name_helpers():
    from src.cli import _slug, _unique_dir

    assert _slug("Hello, World: part 2!") == "Hello-World-part-2"
    assert _slug("你好，世界") == "你好-世界"
    assert _slug("???") == "video"  # nothing usable left
    assert len(_slug("x" * 200)) == 60
    assert _slug("a" * 59 + " b") == "a" * 59  # no dangling separator after truncating


def test_unique_dir_never_overwrites(tmp_path):
    from src.cli import _unique_dir

    (tmp_path / "01-a").mkdir()
    (tmp_path / "01-a-2").mkdir()
    assert _unique_dir(str(tmp_path), "01-a") == str(tmp_path / "01-a-3")
    assert _unique_dir(str(tmp_path), "02-b") == str(tmp_path / "02-b")


def test_no_url_is_an_error(monkeypatch):
    _stub_pipeline_named(monkeypatch)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2 and "at least one video URL" in result.output


def test_an_unreadable_file_is_a_clear_error(monkeypatch, tmp_path):
    _stub_pipeline_named(monkeypatch)
    result = runner.invoke(app, ["run", "--file", str(tmp_path / "missing.txt")])
    assert result.exit_code == 2 and "cannot read" in result.output
