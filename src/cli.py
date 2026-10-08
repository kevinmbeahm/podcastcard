"""Typer CLI for the Mandarin podcast vocabulary study tool."""

from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="podcastcard",
    help="Extract and study Mandarin vocabulary from podcast audio.",
    add_completion=False,
)
console = Console()


def _progress() -> Progress:
    """A spinner + bar + percentage + elapsed time (the bar pulses until a total is known)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    )


@app.callback()
def _main() -> None:
    """Extract and study Mandarin vocabulary from podcast audio."""
    # A callback keeps Typer in multi-command mode, so `podcastcard run <url>`
    # works. With a single command and no callback, Typer collapses it and
    # treats "run" as the URL argument.


def _parse_hsk_filter(hsk_levels: str) -> set[int] | None:
    """
    Parse --hsk-levels option.

    Returns None for "all", or a set of integer levels otherwise.
    """
    if hsk_levels.strip().lower() == "all":
        return None
    parts = [p.strip() for p in hsk_levels.split(",") if p.strip()]
    levels: set[int] = set()
    for p in parts:
        try:
            lvl = int(p)
            if lvl < 0 or lvl > 6:
                raise ValueError
            levels.add(lvl)
        except ValueError:
            raise typer.BadParameter(
                f"Invalid HSK level '{p}'. Must be integers 0-6 or 'all'."
            )
    return levels


def _slug(text: str, limit: int = 60) -> str:
    """A safe, readable folder name (Chinese titles stay Chinese)."""
    cleaned = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE).strip("-")
    return cleaned[:limit].rstrip("-") or "video"


def _unique_dir(parent: str, name: str) -> str:
    """parent/name, or parent/name-2, parent/name-3 ... if that already exists."""
    candidate, n = os.path.join(parent, name), 2
    while os.path.exists(candidate):
        candidate = os.path.join(parent, f"{name}-{n}")
        n += 1
    return candidate


def _collect_urls(urls: Optional[list[str]], file: Optional[Path]) -> list[str]:
    """URLs given on the command line (taken as they are) plus those listed in --file."""
    from .jobs import parse_urls

    collected = [u.strip() for u in (urls or []) if u.strip()]
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            raise typer.BadParameter(f"cannot read {file}: {exc}", param_hint="--file")
        listed, rejected = parse_urls(text)
        for item in rejected:
            console.print(f"[yellow]Skipping[/yellow] {item['line']!r}: {item['reason']}")
        collected += listed
    collected = list(dict.fromkeys(collected))  # a link given twice is processed once
    if not collected:
        raise typer.BadParameter("give at least one video URL, or --file with a list of them.")
    return collected


def _analyze(
    url: str,
    output_dir: str,
    *,
    model: str,
    device: str,
    level_filter: set[int] | None,
    anki: bool,
    show_words: bool,
    folder_for: Callable[[str], str] | None = None,
) -> dict:
    """The whole pipeline for one video. Returns ``{"title", "folder", "words"}``.

    With ``folder_for`` the files are first downloaded into ``output_dir`` and then that
    folder is renamed to ``folder_for(audio_path)`` once the video's title is known.
    FFmpegNotFoundError is left to the caller; any other failure propagates too.
    """
    # Import here so startup is fast and errors surface only when needed
    from .audio import download_audio
    from .extract import extract_words
    from .transcribe import loading_message, transcribe
    from .transcript import format_text, format_vtt

    # ── Step 1: Download ────────────────────────────────────────────────────
    with _progress() as progress:
        task = progress.add_task("Downloading audio…", total=None)

        def on_download(fraction: float) -> None:
            progress.update(
                task,
                total=100,
                completed=fraction * 100,
                description="Converting audio to mp3…" if fraction >= 1 else "Downloading audio…",
            )

        audio_path = download_audio(url, output_dir, on_progress=on_download)

    if folder_for is not None:
        final_dir = folder_for(audio_path)
        os.replace(output_dir, final_dir)
        audio_path = os.path.join(final_dir, os.path.basename(audio_path))
        output_dir = final_dir
    title = os.path.splitext(os.path.basename(audio_path))[0]
    console.print(f"[green]✓[/green] Audio saved to [bold]{audio_path}[/bold]")

    # ── Step 2: Transcribe ──────────────────────────────────────────────────
    with _progress() as progress:
        task = progress.add_task(loading_message(model), total=None)

        def on_transcribe(done: float, total: float) -> None:
            progress.update(
                task,
                total=total,
                completed=done,
                description=f"Transcribing with Whisper [{model}]…",
            )

        def on_status(text: str, fraction: float | None = None) -> None:
            progress.update(task, description=text, completed=(fraction or 0) * 100, total=100)

        segments = transcribe(
            audio_path,
            model_size=model,
            device=device,
            on_progress=on_transcribe,
            on_status=on_status,
        )

    console.print(f"[green]✓[/green] Transcribed {len(segments)} segments.")

    vtt_path = os.path.join(output_dir, "transcript.vtt")
    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(vtt_path, "w", encoding="utf-8") as fh:
        fh.write(format_vtt(segments))
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(format_text(segments))
    console.print(f"[green]✓[/green] Transcript saved to [bold]{txt_path}[/bold] (+ .vtt)")

    # ── Step 3: Extract vocabulary ──────────────────────────────────────────
    words = extract_words(segments)
    console.print(f"[green]✓[/green] Extracted {len(words)} unique words.")

    # ── Step 4: Filter ──────────────────────────────────────────────────────
    if level_filter is not None:
        words = [w for w in words if w.hsk_level in level_filter]
        console.print(
            f"[yellow]→[/yellow] {len(words)} words match HSK level(s): "
            + ", ".join(str(l) for l in sorted(level_filter))
        )

    if not words:
        console.print("[bold red]No words matched the filter.[/bold red]")
        return {"title": title, "folder": output_dir, "words": 0}

    # ── Step 5: Display ─────────────────────────────────────────────────────
    if show_words:
        _display_words(words)

    # ── Step 6: Export CSV (and Anki) ───────────────────────────────────────
    csv_path = os.path.join(output_dir, "words.csv")
    _export_csv(words, csv_path)
    console.print(f"\n[green]✓[/green] Exported [bold]{csv_path}[/bold]")

    if anki:
        from dataclasses import asdict

        from .anki import write_apkg

        apkg_path = os.path.join(output_dir, "anki_deck.apkg")
        deck_title = title.replace("::", ":")
        count = write_apkg([asdict(w) for w in words], apkg_path, f"PodcastCard::{deck_title}", source=deck_title)
        console.print(f"[green]✓[/green] Anki deck with {count} cards: [bold]{apkg_path}[/bold]")

    return {"title": title, "folder": output_dir, "words": len(words)}


@app.command()
def run(
    urls: Optional[list[str]] = typer.Argument(None, help="URL(s) of the podcasts/videos to process."),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="A text file with one URL per line (blank lines and lines starting with # are ignored).",
    ),
    model: str = typer.Option("base", "--model", help="Whisper model size (tiny/base/small/medium/large-v2)."),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Where to run Whisper: auto (GPU if it works, else CPU), cpu, or cuda.",
    ),
    hsk_levels: str = typer.Option(
        "all",
        "--hsk-levels",
        help="Comma-separated HSK levels to display, e.g. '4,5,6'. Use '0' for unknown words. Default: all.",
    ),
    output: str = typer.Option("./output", "--output", help="Directory for output files."),
    anki: bool = typer.Option(
        False, "--anki", help="Also write an Anki deck (anki_deck.apkg) for the displayed words."
    ),
) -> None:
    """
    Full pipeline: download audio → transcribe → extract vocabulary → display & export.

    Writes transcript.vtt and transcript.txt (the full transcript) and words.csv
    to the output directory.

    Give several URLs (or --file) to process a batch, one after another: each video then gets its
    own folder inside the output directory (01-title, 02-title, ...), a failure in one does not
    stop the rest, and a summary is printed at the end.
    """
    from .audio import FFmpegNotFoundError

    if device not in ("auto", "cpu", "cuda"):
        raise typer.BadParameter("must be auto, cpu, or cuda", param_hint="--device")
    level_filter = _parse_hsk_filter(hsk_levels)
    all_urls = _collect_urls(urls, file)
    output_dir = os.path.abspath(output)
    os.makedirs(output_dir, exist_ok=True)
    options = dict(model=model, device=device, level_filter=level_filter, anki=anki)

    if len(all_urls) == 1:  # the original behaviour: files go straight into the output directory
        try:
            _analyze(all_urls[0], output_dir, show_words=True, **options)
        except FFmpegNotFoundError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)
        return

    results: list[dict] = []
    total = len(all_urls)
    for index, url in enumerate(all_urls, 1):
        console.rule(f"[bold]Video {index} of {total}[/bold]  {url}")
        staging = os.path.join(output_dir, f".working-{index:02d}")
        os.makedirs(staging, exist_ok=True)
        try:
            result = _analyze(
                url,
                staging,
                show_words=False,
                folder_for=lambda audio, i=index: _unique_dir(
                    output_dir, f"{i:02d}-{_slug(os.path.splitext(os.path.basename(audio))[0])}"
                ),
                **options,
            )
            results.append({"url": url, "ok": True, **result})
        except FFmpegNotFoundError as exc:  # every video would fail the same way
            shutil.rmtree(staging, ignore_errors=True)
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            console.print(f"[bold red]✗ Failed:[/bold red] {exc}")
            results.append({"url": url, "ok": False, "error": str(exc)})

    summary = Table(title="Batch summary", show_lines=False)
    for column in ("#", "Video", "Result", "Folder"):
        summary.add_column(column, overflow="fold")
    for index, item in enumerate(results, 1):
        if item["ok"]:
            summary.add_row(str(index), item["title"], f"[green]✓ {item['words']} words[/green]", item["folder"])
        else:
            summary.add_row(str(index), item["url"], f"[red]✗ {item['error']}[/red]", "")
    console.print()
    console.print(summary)
    failed = sum(1 for item in results if not item["ok"])
    console.print(f"{total - failed} of {total} videos done" + (f", [red]{failed} failed[/red]" if failed else "") + ".")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to listen on."),
    port: int = typer.Option(8000, "--port", help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes (development)."),
) -> None:
    """Start the web app (then open http://localhost:8000)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[bold red]The web server needs uvicorn: pip install -r requirements.txt[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"PodcastCard is running at [bold]http://{'localhost' if host == '127.0.0.1' else host}:{port}[/bold]  (Ctrl+C to stop)")
    uvicorn.run("src.app:app", host=host, port=port, reload=reload)


def _display_words(words: list) -> None:
    """Render words grouped by HSK level using rich."""
    from collections import defaultdict

    grouped: dict[int, list] = defaultdict(list)
    for w in words:
        grouped[w.hsk_level].append(w)

    level_order = sorted(grouped.keys(), key=lambda l: (l == 0, l))

    for level in level_order:
        level_label = f"HSK {level}" if level > 0 else "Unknown (non-HSK)"
        console.rule(f"[bold magenta]{level_label}[/bold magenta]")

        for w in grouped[level]:
            table = Table.grid(padding=(0, 1))
            table.add_column(style="bold yellow", no_wrap=True)
            table.add_column(style="cyan")
            table.add_column(style="dim")
            table.add_row(w.word, w.pinyin, f"HSK {w.hsk_level}" if w.hsk_level else "—")
            console.print(table)
            if w.definition:
                console.print(f"  [green]{w.definition}[/green]")

            for i, ctx in enumerate(w.contexts, 1):
                console.print(f"  [dim]{i}.[/dim] {ctx}")
            console.print()


def _export_csv(words: list, path: str) -> None:
    """Write vocabulary to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["word", "pinyin", "definition", "hsk_level", "frequency", "contexts"],
        )
        writer.writeheader()
        for w in words:
            writer.writerow(
                {
                    "word": w.word,
                    "pinyin": w.pinyin,
                    "definition": w.definition,
                    "hsk_level": w.hsk_level,
                    "frequency": len(w.contexts),
                    "contexts": " | ".join(w.contexts),
                }
            )


if __name__ == "__main__":
    app()
