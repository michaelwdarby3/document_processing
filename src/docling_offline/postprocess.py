"""Helpers for post-processing Docling outputs (e.g., exporting tables)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def export_tables_to_csv(doc_json: Path, output_dir: Path) -> List[Path]:
    """Extract every table from a Docling JSON payload into standalone CSV files.

    Args:
        doc_json: Path to the JSON document produced by `--format json`.
        output_dir: Directory where CSV files should be written.

    Returns:
        List of CSV file paths that were created.
    """

    data = _load_docling_json(doc_json)
    tables: Sequence[Dict[str, Any]] = data.get("tables") or []
    if not tables:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for idx, table in enumerate(tables, start=1):
        rows = _table_rows(table)
        if not rows:
            continue
        csv_name = f"{doc_json.stem}_table_{idx:02d}.csv"
        csv_path = output_dir / csv_name
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in rows:
                writer.writerow(row)
        written.append(csv_path)

    return written


# -- internal helpers -----------------------------------------------------


def _load_docling_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _table_rows(table: Dict[str, Any]) -> List[List[str]]:
    grid: Iterable[Iterable[Any]] = table.get("data", {}).get("grid") or []
    rows: List[List[str]] = []
    for row in grid:
        row_cells: List[str] = []
        for cell in row:
            text = _cell_text(cell)
            if text:
                if isinstance(cell, dict):
                    if cell.get("row_section"):
                        text = f"[SECTION] {text}"
                    elif cell.get("row_header"):
                        text = f"[ROW HEADER] {text}"
                    elif cell.get("column_header"):
                        text = f"[COLUMN HEADER] {text}"
            row_cells.append(text)
        rows.append(row_cells)
    return rows


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        raw = cell.get("text") or ""
    else:
        raw = str(cell) if cell is not None else ""
    return " ".join(raw.split())


__all__ = ["export_tables_to_csv"]
