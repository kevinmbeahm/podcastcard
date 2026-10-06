"""CLI wiring tests (pipeline stages are stubbed; no network or audio needed)."""

import sys
import types

from typer.testing import CliRunner

from src.cli import app
from src.extract import Segment

runner = CliRunner()


def _stub_pipeline(monkeypatch):
    audio = types.ModuleType("src.audio")
    audio.download_audio = lambda url, out: f"{out}/fake.mp3"
    audio.FFmpegNotFoundError = type("FFmpegNotFoundError", (RuntimeError,), {})
    transcribe = types.ModuleType("src.transcribe")
    transcribe.transcribe = lambda path, model_size="base", device="auto": [
        Segment(0, 3, "我喜欢听中文播客，学习新的词汇。")
    ]
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
