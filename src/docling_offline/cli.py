"""CLI entrypoint for docling-offline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List

import typer
from rich.console import Console
from rich.table import Table

from .config import (
    ConfigError,
    ConvertConfig,
    PrefetchConfig,
    SUPPORTED_FORMATS,
    SUPPORTED_OCR_ENGINES,
    format_list,
)
from .processor import ConversionResult, convert_documents, prefetch_models, summarize_results

console = Console()
app = typer.Typer(help="Offline-first PDF conversion tool powered by Docling.")


def _handle_config_error(error: ConfigError) -> None:
    console.print(f"[bold red]configuration error:[/bold red] {error}")
    raise typer.Exit(code=2)


@app.command("prefetch-models")
def prefetch_models_command(
    artifacts_path: Path = typer.Option(
        ".artifacts/",
        "--artifacts-path",
        help="Directory where Docling artifacts are stored for offline use.",
    ),
    device: str = typer.Option("auto", "--device", help="Execution device: auto|cuda|mps|cpu."),
    threads: int = typer.Option(4, "--threads", min=1, help="Thread hint for Docling."),
) -> None:
    """Download and cache Docling artifacts locally."""

    try:
        config = PrefetchConfig(
            artifacts_path=artifacts_path,
            device=device,
            threads=threads,
        )
    except ConfigError as error:
        _handle_config_error(error)
        return

    console.print(
        f"Prefetching Docling models into [bold]{config.artifacts_path}[/bold] "
        f"(device=[bold]{config.device}[/bold], threads={config.threads})..."
    )
    try:
        prefetch_models(config)
    except Exception as exc:  # pragma: no cover - runtime dependency
        console.print(f"[bold red]prefetch failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print("[bold green]Artifacts ready for offline usage.[/bold green]")


@app.command("convert")
def convert_command(
    inputs: List[Path] = typer.Argument(..., help="PDF files or directories to convert. Supports globs."),
    output: Path = typer.Option(
        Path("out/"),
        "--output",
        "-o",
        help="Directory for exported documents.",
    ),
    formats: List[str] = typer.Option(
        ["json"],
        "--format",
        "-f",
        help=f"Output formats (choices: {', '.join(sorted(SUPPORTED_FORMATS))}). Repeat for multiple.",
    ),
    artifacts_path: Path = typer.Option(
        ".artifacts/",
        "--artifacts-path",
        help="Docling artifacts directory populated via `prefetch-models`.",
    ),
    device: str = typer.Option("auto", "--device", help="Execution device: auto|cuda|mps|cpu."),
    threads: int = typer.Option(4, "--threads", min=1, help="Thread hint for Docling."),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        help="File-level processes for parallel conversion. Start small on laptops.",
    ),
    layout_batch_size: int = typer.Option(4, "--layout-batch-size", min=1),
    ocr_batch_size: int = typer.Option(4, "--ocr-batch-size", min=1),
    table_batch_size: int = typer.Option(4, "--table-batch-size", min=1),
    ocr: str = typer.Option(
        "none",
        "--ocr",
        help=f"OCR mode ({', '.join(sorted(SUPPORTED_OCR_ENGINES))}).",
    ),
    ocr_langs: str = typer.Option("en", "--ocr-langs", help="Comma-separated OCR languages."),
    force_full_page_ocr: bool = typer.Option(
        False,
        "--force-full-page-ocr",
        help="Force OCR over every page (slower, but necessary for poor scans).",
    ),
    generate_page_images: bool = typer.Option(
        False,
        "--generate-page-images",
        help="Emit page-level images alongside HTML output.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on the first failure and exit with code 1.",
    ),
) -> None:
    """Convert one or more PDFs into offline-friendly formats."""

    languages = tuple(part.strip() for part in ocr_langs.split(","))

    try:
        config = ConvertConfig(
            inputs=tuple(inputs),
            output=output,
            formats=tuple(formats),
            artifacts_path=artifacts_path,
            device=device,
            threads=threads,
            workers=workers,
            layout_batch_size=layout_batch_size,
            ocr_batch_size=ocr_batch_size,
            table_batch_size=table_batch_size,
            ocr=ocr,
            ocr_langs=languages,
            force_full_page_ocr=force_full_page_ocr,
            generate_page_images=generate_page_images,
            fail_fast=fail_fast,
        )
    except ConfigError as error:
        _handle_config_error(error)
        return

    if config.workers > 1:
        console.print(
            "[yellow]Warning:[/] Multiple workers increase RAM/VRAM usage. "
            "Ensure your machine has enough headroom."
        )

    start_time = perf_counter()
    overall_timer = console.status("Starting conversion...", spinner="dots")
    overall_timer.start()

    console.print(
        f"Converting {len(config.discovered_inputs)} PDFs "
        f"into [bold]{config.output}[/bold] "
        f"({format_list(config.formats)})."
    )

    try:
        results = convert_documents(config)
    except Exception as exc:  # pragma: no cover - runtime dependency
        console.print(f"[bold red]conversion failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    finally:
        overall_timer.stop()
    elapsed = perf_counter() - start_time

    _render_results(results, config.output)

    stats = summarize_results(results)
    console.print(
        f"[bold]{stats['succeeded']}[/bold] succeeded, [bold]{stats['failed']}[/bold] failed."
    )
    console.print(f"[dim]Elapsed time: {elapsed:.2f}s[/dim]")
    if stats["failed"] > 0 and config.fail_fast:
        raise typer.Exit(code=1)


def _render_results(results: Sequence[ConversionResult], output_dir: Path) -> None:
    table = Table(title="Conversion summary")
    table.add_column("Source", overflow="fold")
    table.add_column("Status", style="bold")
    table.add_column("Outputs", overflow="fold")
    table.add_column("Message", overflow="fold")

    for result in sorted(results, key=lambda item: item.source):
        status = "[green]ok[/green]" if result.success else "[red]failed[/red]"
        output_strings = []
        for path in result.outputs:
            try:
                display = path.relative_to(output_dir)
            except ValueError:
                display = path
            output_strings.append(str(display))
        outputs = "\n".join(output_strings)
        table.add_row(str(result.source), status, outputs, result.message)

    console.print(table)


def main() -> None:
    """CLI entrypoint for `python -m docling_offline.cli`."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
