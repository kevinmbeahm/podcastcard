"""Typer CLI for the Mandarin podcast vocabulary study tool."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="podcastcard",
    help="Extract and study Mandarin vocabulary from podcast audio.",
    add_completion=False,
)
console = Console()


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
    hsk_levels: str = typer.Option(
        "all",
        "--hsk-levels",
        help="Comma-separated HSK levels to display, e.g. '4,5,6'. Use '0' for unknown words. Default: all.",
    ),
    output: str = typer.Option("./output", "--output", help="Directory for output files."),
) -> None:
    """
    Full pipeline: download audio → transcribe → extract vocabulary → display & export.
    """
    # Import here so startup is fast and errors surface only when needed
    from .audio import download_audio
    from .transcribe import transcribe
    from .extract import extract_words, WordOccurrence

    level_filter = _parse_hsk_filter(hsk_levels)
    output_dir = os.path.abspath(output)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Download ────────────────────────────────────────────────────
    audio_path: str
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Downloading audio…", total=None)
        audio_path = download_audio(url, output_dir)

    console.print(f"[green]✓[/green] Audio saved to [bold]{audio_path}[/bold]")

    # ── Step 2: Transcribe ──────────────────────────────────────────────────
    from .extract import Segment  # noqa: F401 (already imported via extract_words)

    segments: list
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(f"Transcribing with Whisper [{model}]…", total=None)
        segments = transcribe(audio_path, model_size=model)

    console.print(f"[green]✓[/green] Transcribed {len(segments)} segments.")

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

            for i, ctx in enumerate(w.contexts, 1):
                console.print(f"  [dim]{i}.[/dim] {ctx}")
            console.print()


def _export_csv(words: list, path: str) -> None:
    """Write vocabulary to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["word", "pinyin", "hsk_level", "frequency", "contexts"],
        )
        writer.writeheader()
        for w in words:
            writer.writerow(
                {
                    "word": w.word,
                    "pinyin": w.pinyin,
                    "hsk_level": w.hsk_level,
                    "frequency": len(w.contexts),
                    "contexts": " | ".join(w.contexts),
                }
            )


if __name__ == "__main__":
    app()
