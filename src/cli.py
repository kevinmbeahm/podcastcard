"""Typer CLI for the Mandarin podcast vocabulary study tool."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

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


@app.command()
def run(
    url: str = typer.Argument(..., help="URL of the podcast/video to process."),
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
    """
    # Import here so startup is fast and errors surface only when needed
    from .audio import FFmpegNotFoundError, download_audio
    from .transcribe import loading_message, transcribe
    from .extract import extract_words, WordOccurrence

    if device not in ("auto", "cpu", "cuda"):
        raise typer.BadParameter("must be auto, cpu, or cuda", param_hint="--device")
    level_filter = _parse_hsk_filter(hsk_levels)
    output_dir = os.path.abspath(output)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Download ────────────────────────────────────────────────────
    audio_path: str
    with _progress() as progress:
        task = progress.add_task("Downloading audio…", total=None)

        def on_download(fraction: float) -> None:
            progress.update(
                task,
                total=100,
                completed=fraction * 100,
                description="Converting audio to mp3…" if fraction >= 1 else "Downloading audio…",
            )

        try:
            audio_path = download_audio(url, output_dir, on_progress=on_download)
        except FFmpegNotFoundError as exc:
            progress.stop()
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Audio saved to [bold]{audio_path}[/bold]")

    # ── Step 2: Transcribe ──────────────────────────────────────────────────
    from .extract import Segment  # noqa: F401 (already imported via extract_words)

    segments: list
    with _progress() as progress:
        task = progress.add_task(loading_message(model), total=None)

        def on_transcribe(done: float, total: float) -> None:
            progress.update(
                task,
                total=total,
                completed=done,
                description=f"Transcribing with Whisper [{model}]…",
            )

        segments = transcribe(audio_path, model_size=model, device=device, on_progress=on_transcribe)

    console.print(f"[green]✓[/green] Transcribed {len(segments)} segments.")

    from .transcript import format_text, format_vtt

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
        console.print("[bold red]No words matched the filter. Exiting.[/bold red]")
        raise typer.Exit(code=0)

    # ── Step 5: Display ─────────────────────────────────────────────────────
    _display_words(words)

    # ── Step 6: Export CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "words.csv")
    _export_csv(words, csv_path)
    console.print(f"\n[green]✓[/green] Exported [bold]{csv_path}[/bold]")

    if anki:
        from dataclasses import asdict

        from .anki import write_apkg

        apkg_path = os.path.join(output_dir, "anki_deck.apkg")
        title = os.path.splitext(os.path.basename(audio_path))[0].replace("::", ":")
        count = write_apkg([asdict(w) for w in words], apkg_path, f"PodcastCard::{title}", source=title)
        console.print(f"[green]✓[/green] Anki deck with {count} cards: [bold]{apkg_path}[/bold]")


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
