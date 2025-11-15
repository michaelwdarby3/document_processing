#!/usr/bin/env python3
"""Visualize Docling table bounding boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pypdfium2 as pdfium


def show_table_boxes(
    pdf_path: Path,
    json_path: Path,
    table_index: int = 0,
    page_index: int = 0,
    output: Optional[Path] = None,
) -> Optional[Path]:
    with json_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    tables = doc.get("tables") or []
    if not tables:
        raise ValueError("No tables found in JSON.")
    if table_index >= len(tables) or table_index < 0:
        raise IndexError(f"Table index {table_index} out of range (0-{len(tables)-1}).")

    table = tables[table_index]
    data = table.get("data", {}) or {}
    grid = data.get("grid") or []
    table_cells = data.get("table_cells")
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index]

    page_width = page.get_width()
    page_height = page.get_height()
    render = page.render(scale=1.5)
    pil_image = render.to_pil()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(pil_image)
    scale_x = render.width / page_width
    scale_y = render.height / page_height

    if table_cells:
        iter_cells = table_cells
    else:
        iter_cells = [cell for row in grid for cell in row]

    for cell in iter_cells:
            bbox = cell.get("bbox", {})
            if not bbox:
                continue
            origin = (bbox.get("coord_origin") or "BOTTOMLEFT").upper()
            x = bbox["l"] * scale_x
            width = (bbox["r"] - bbox["l"]) * scale_x
            if origin == "TOPLEFT":
                y = bbox["t"] * scale_y
                height = (bbox["b"] - bbox["t"]) * scale_y
            else:
                y = render.height - bbox["t"] * scale_y
                height = (bbox["t"] - bbox["b"]) * scale_y
            if width <= 0 or height <= 0:
                continue
            if cell.get("row_section"):
                color = "orange"
            elif cell.get("column_header"):
                color = "red"
            elif cell.get("row_header"):
                color = "green"
            else:
                color = "blue"
            ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1))
            snippet = (cell.get("text") or "")[:20]
            snippet = snippet.replace("\n", " ").replace("\r", " ").replace("$", r"\$")
            ax.text(x, y, snippet, fontsize=6, color=color, usetex=False)

    ax.set_title(f"Table {table_index} on page {page_index}")
    backend = matplotlib.get_backend().lower()
    saved_path: Optional[Path] = None
    should_save = output is not None or "agg" in backend

    if should_save:
        saved_path = output or Path(f"{json_path.stem}_table_{table_index:02d}.png")
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saved_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Docling table bounding boxes.")
    parser.add_argument("pdf", type=Path, help="Path to the source PDF")
    parser.add_argument("json", type=Path, help="Path to the Docling JSON output")
    parser.add_argument("--table", type=int, default=0, help="Zero-based table index to show")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index (default 0)")
    parser.add_argument("--output", type=Path, help="Optional path to save the visualization as an image")
    args = parser.parse_args()

    saved = show_table_boxes(
        args.pdf,
        args.json,
        table_index=args.table,
        page_index=args.page,
        output=args.output,
    )
    if saved:
        print(f"Saved visualization to {saved.resolve()}")


if __name__ == "__main__":
    main()
