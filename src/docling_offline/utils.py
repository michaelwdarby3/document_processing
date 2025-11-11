from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import shutil

try:  # pragma: no cover - docling_core is a runtime dependency
    from docling_core.types.doc import TableItem
    from docling_core.types.doc.base import BoundingBox, CoordOrigin
    from docling_core.types.doc.document import DoclingDocument
except ImportError:  # pragma: no cover
    TableItem = Any  # type: ignore
    BoundingBox = Any  # type: ignore
    DoclingDocument = Any  # type: ignore
    CoordOrigin = Any  # type: ignore


def ensure_tableformer_structure(artifacts_path: Path) -> None:
    tableformer_root = artifacts_path / "model_artifacts" / "tableformer"
    if not tableformer_root.exists():
        return

    _ensure_mode(tableformer_root, "fast", ["fast", "fat"])
    _ensure_mode(tableformer_root, "accurate", ["accurate", "acr"])


def clip_overlapping_table_cells(document: Any, gutter: float = 1.0) -> Tuple[int, Dict[int, int]]:
    """Remove stray TableFormer cells that spill into neighbouring tables.

    When table cell matching is disabled (--no-table-cell-matching), the TableFormer
    grid may include cells that belong to an adjacent table (because Docling's table
    bounding boxes can overlap horizontally). This helper clips each table's cells so
    that their horizontal centres stay inside the non-overlapping portion of the table
    bounding box.

    Args:
        document: DoclingDocument produced by the converter.
        gutter: Extra padding (in PDF units) to keep between neighbouring tables.

    Returns:
        Tuple of (total removed cells, mapping of table ids to removed counts).
    """

    tables: Sequence[TableItem] = getattr(document, "tables", None) or []
    pages: Any = getattr(document, "pages", None)
    if not tables or pages is None:
        return 0, {}

    tables_by_page: dict[int, List[Tuple[TableItem, BoundingBox]]] = defaultdict(list)
    for table in tables:
        bbox = _table_bbox(table)
        page_no = _table_page(table)
        if bbox is None or page_no is None:
            continue
        tables_by_page[page_no].append((table, bbox))

    removed = 0
    removed_by_table: Dict[int, int] = {}
    for page_tables in tables_by_page.values():
        page_tables.sort(key=lambda item: _normalize_bounds(item[1])[0])
        for idx, (table, bbox) in enumerate(page_tables):
            bounds = _normalize_bounds(bbox)
            clip_left = bounds[0]
            for prev_idx in range(idx):
                prev_table, prev_bbox = page_tables[prev_idx]
                prev_bounds = _normalize_bounds(prev_bbox)
                if (
                    _vertical_overlap(prev_bounds, bounds)
                    and prev_bounds[1] > clip_left
                ):
                    clip_left = max(clip_left, prev_bounds[1] + gutter)
            if clip_left > bounds[0]:
                trimmed = _clip_cells_left_of(table, clip_left)
                if trimmed:
                    removed += trimmed
                    removed_by_table[id(table)] = removed_by_table.get(id(table), 0) + trimmed
    return removed, removed_by_table


def sanitize_table_bboxes(document: Any) -> Dict[int, dict[str, float]]:
    """Clamp all table bounding boxes to their page dimensions."""

    page_meta = _page_metadata(document)
    tables: Sequence[TableItem] = getattr(document, "tables", None) or []
    for table in tables:
        page_no = _table_page(table)
        stats = page_meta.get(page_no) or {}
        page_w = stats.get("width")
        page_h = stats.get("height")
        if page_w is None or page_h is None:
            continue
        data = getattr(table, "data", None)
        cells: Sequence[Any] = getattr(data, "table_cells", None) or []
        for cell in cells:
            bbox = getattr(cell, "bbox", None)
            if bbox is None:
                continue
            _clamp_bbox(bbox, page_w, page_h)
    return page_meta


def _table_bbox(table: TableItem) -> Optional[BoundingBox]:
    prov = getattr(table, "prov", None) or []
    if prov:
        entry = prov[0]
        return getattr(entry, "bbox", None)
    cluster = getattr(table, "cluster", None)
    if cluster is not None:
        return getattr(cluster, "bbox", None)
    return None


def _table_page(table: TableItem) -> Optional[int]:
    prov = getattr(table, "prov", None) or []
    if prov:
        entry = prov[0]
        return getattr(entry, "page_no", None)
    return None


def _normalize_bounds(bbox: BoundingBox) -> Tuple[float, float, float, float]:
    left = min(bbox.l, bbox.r)
    right = max(bbox.l, bbox.r)
    low = min(bbox.t, bbox.b)
    high = max(bbox.t, bbox.b)
    return left, right, low, high


def _vertical_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], min_overlap: float = 10.0) -> bool:
    low = max(a[2], b[2])
    high = min(a[3], b[3])
    return high - low > min_overlap


def _clip_cells_left_of(table: TableItem, min_x: float) -> int:
    data = getattr(table, "data", None)
    if data is None:
        return 0
    cells: List[Any] = getattr(data, "table_cells", []) or []
    if not cells:
        return 0

    kept: List[Any] = []
    removed = 0
    for cell in cells:
        bbox = getattr(cell, "bbox", None)
        if bbox is None:
            kept.append(cell)
            continue
        left = min(bbox.l, bbox.r)
        right = max(bbox.l, bbox.r)
        center_x = (left + right) / 2
        if center_x < min_x:
            removed += 1
            continue
        kept.append(cell)

    if removed:
        data.table_cells = kept
        _reindex_table_columns(data)
    return removed


def _reindex_table_columns(table_data: Any) -> None:
    cells: List[Any] = getattr(table_data, "table_cells", []) or []
    if not cells:
        table_data.num_cols = 0
        return

    unique_starts = sorted({cell.start_col_offset_idx for cell in cells})
    index_map = {old: idx for idx, old in enumerate(unique_starts)}
    max_end = 0

    for cell in cells:
        width = max(1, cell.end_col_offset_idx - cell.start_col_offset_idx)
        new_start = index_map[cell.start_col_offset_idx]
        cell.start_col_offset_idx = new_start
        cell.end_col_offset_idx = new_start + width
        max_end = max(max_end, cell.end_col_offset_idx)

    table_data.num_cols = max_end


def write_table_metadata(
    document: Any,
    target: Path,
    *,
    clipping_enabled: bool,
    clipped_cells_total: int,
    clipped_by_table: Optional[Dict[int, int]] = None,
    table_mode: str = "",
    table_cell_matching: bool = True,
    formats: Sequence[str] | None = None,
    page_metadata: Optional[Dict[int, dict[str, float]]] = None,
) -> Path:
    """Persist a detailed summary of every table in the document."""

    tables: Sequence[TableItem] = getattr(document, "tables", None) or []
    pages_meta = page_metadata or _page_metadata(document)
    doc_meta = {
        "name": getattr(document, "name", None),
        "version": getattr(document, "version", None),
        "page_count": len(pages_meta),
    }

    clipped_by_table = clipped_by_table or {}
    table_entries: List[dict[str, Any]] = []
    tables_with_clipping = 0

    for idx, table in enumerate(tables):
        data = getattr(table, "data", None)
        bbox = _table_bbox(table)
        page_no = _table_page(table)
        page_stats = pages_meta.get(page_no)
        table_stats = _table_statistics(table, data, bbox, page_stats)
        clipped = clipped_by_table.get(id(table), 0)
        if clipped:
            tables_with_clipping += 1

        table_entries.append(
            {
                "index": idx,
                "label": getattr(table, "label", None),
                "page": page_no,
                "self_ref": getattr(table, "self_ref", None),
                "bbox": _bbox_to_dict(bbox),
                "metrics": table_stats,
                "clipped_cells": clipped,
            }
        )

    summary = {
        "document": doc_meta,
        "table_processing": {
            "table_mode": table_mode,
            "table_cell_matching": table_cell_matching,
            "clip_table_overlap": clipping_enabled,
            "clipped_cells_total": clipped_cells_total,
            "tables_with_clipping": tables_with_clipping,
            "exported_formats": sorted(set(formats or [])),
        },
        "tables": table_entries,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _bbox_to_dict(bbox: Optional[BoundingBox]) -> Optional[dict[str, float]]:
    if bbox is None:
        return None
    return {
        "l": bbox.l,
        "r": bbox.r,
        "t": bbox.t,
        "b": bbox.b,
        "coord_origin": str(getattr(bbox, "coord_origin", "")),
    }


def _clamp_bbox(bbox: BoundingBox, page_w: float, page_h: float) -> None:
    def clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    bbox.l = clamp(bbox.l, 0.0, page_w)
    bbox.r = clamp(bbox.r, 0.0, page_w)
    if bbox.coord_origin == CoordOrigin.TOPLEFT:
        bbox.t = clamp(bbox.t, 0.0, page_h)
        bbox.b = clamp(bbox.b, 0.0, page_h)
        if bbox.b < bbox.t:
            bbox.t, bbox.b = min(bbox.t, bbox.b), max(bbox.t, bbox.b)
    else:
        bbox.b = clamp(bbox.b, 0.0, page_h)
        bbox.t = clamp(bbox.t, 0.0, page_h)
        if bbox.t < bbox.b:
            bbox.b, bbox.t = min(bbox.b, bbox.t), max(bbox.b, bbox.t)


def _page_metadata(document: Any) -> Dict[int, dict[str, float]]:
    pages_meta: Dict[int, dict[str, float]] = {}
    pages = getattr(document, "pages", {}) or {}
    for key, page in pages.items():
        try:
            page_no = int(key)
        except Exception:
            page_no = key
        size = getattr(page, "size", None)
        if size is None:
            continue
        width = getattr(size, "width", None)
        height = getattr(size, "height", None)
        if width is None or height is None:
            continue
        pages_meta[page_no] = {
            "width": width,
            "height": height,
            "area": width * height if width and height else None,
        }
    return pages_meta


def _table_statistics(
    table: TableItem,
    data: Any,
    bbox: Optional[BoundingBox],
    page_stats: Optional[dict[str, float]],
) -> dict[str, Any]:
    num_rows = getattr(data, "num_rows", None)
    num_cols = getattr(data, "num_cols", None)
    grid = getattr(data, "grid", None) or []
    cells = getattr(data, "table_cells", None) or []
    total_cells = len(cells) if cells else None
    header_rows = _count_header_rows(grid)
    header_cols = _count_header_columns(grid, num_cols)
    bbox_metrics = _bbox_metrics(bbox, page_stats)
    merged_cells = any(
        getattr(cell, "row_span", 1) > 1 or getattr(cell, "col_span", 1) > 1
        for cell in cells
    )
    density = None
    if num_rows and num_cols and cells:
        density = len(cells) / float(num_rows * num_cols)

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "header_rows": header_rows,
        "header_columns": header_cols,
        "total_cells": total_cells,
        "cells_with_bbox": _count_cells_with_bbox(cells),
        "data_density": density,
        "has_merged_cells": merged_cells,
        "avg_column_widths": _average_column_widths(grid, num_cols),
        "avg_row_heights": _average_row_heights(grid, num_rows),
        "bbox_metrics": bbox_metrics,
    }


def _count_header_rows(grid: Sequence[Sequence[Any]]) -> int:
    count = 0
    for row in grid:
        if any(getattr(cell, "column_header", False) for cell in row):
            count += 1
        else:
            break
    return count


def _count_header_columns(grid: Sequence[Sequence[Any]], num_cols: Optional[int]) -> Optional[int]:
    if not num_cols:
        return None
    header_cols = set()
    for row in grid:
        for idx, cell in enumerate(row):
            if getattr(cell, "column_header", False):
                header_cols.add(idx)
    return len(header_cols) if header_cols else 0


def _count_cells_with_bbox(cells: Sequence[Any]) -> Optional[int]:
    if not cells:
        return None
    return sum(1 for cell in cells if getattr(cell, "bbox", None) is not None)


def _average_column_widths(grid: Sequence[Sequence[Any]], num_cols: Optional[int]) -> Optional[List[Optional[float]]]:
    if not num_cols or not grid:
        return None
    totals = [0.0] * num_cols
    counts = [0] * num_cols
    for row in grid:
        for idx in range(min(len(row), num_cols)):
            bbox = getattr(row[idx], "bbox", None)
            if bbox is None:
                continue
            width = abs(bbox.r - bbox.l)
            totals[idx] += width
            counts[idx] += 1
    averages = []
    for total, count in zip(totals, counts):
        averages.append(total / count if count else None)
    return averages


def _average_row_heights(grid: Sequence[Sequence[Any]], num_rows: Optional[int]) -> Optional[List[Optional[float]]]:
    if not num_rows or not grid:
        return None
    heights = []
    for row in grid:
        row_heights = [
            abs(cell.bbox.t - cell.bbox.b)
            for cell in row
            if getattr(cell, "bbox", None) is not None
        ]
        heights.append(sum(row_heights) / len(row_heights) if row_heights else None)
    return heights if any(h is not None for h in heights) else None


def _bbox_metrics(bbox: Optional[BoundingBox], page_stats: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
    if bbox is None:
        return None
    width = abs(bbox.r - bbox.l)
    height = abs(bbox.t - bbox.b)
    area = width * height
    coverage = None
    if page_stats and page_stats.get("area"):
        coverage = area / page_stats["area"] if page_stats["area"] else None
    return {
        "width": width,
        "height": height,
        "area": area,
        "coverage": coverage,
    }


def export_tables_to_xlsx(document: Any, target: Path) -> Optional[Path]:
    tables: Sequence[TableItem] = getattr(document, "tables", None) or []
    if not tables:
        return None

    try:
        from openpyxl import Workbook
    except ImportError:  # pragma: no cover
        raise RuntimeError("openpyxl is required for Excel export but is not installed.")

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    for idx, table in enumerate(tables, start=1):
        data = getattr(table, "data", None)
        grid = getattr(data, "grid", None)
        sheet = wb.create_sheet(_sheet_name(table, idx))
        if not grid:
            continue
        for row_idx, row in enumerate(grid, start=1):
            for col_idx, cell in enumerate(row, start=1):
                text = ""
                if cell is not None:
                    text = getattr(cell, "text", "")
                sheet.cell(row=row_idx, column=col_idx, value=text)

    target.parent.mkdir(parents=True, exist_ok=True)
    wb.save(target)
    return target


def _sheet_name(table: TableItem, index: int) -> str:
    label = getattr(table, "label", None)
    base = f"Table{index:02d}"
    if label:
        base = f"{base}_{label}"
    # Excel sheet names limited to 31 chars and cannot contain certain characters
    sanitized = "".join(ch for ch in base if ch not in r'[]:*?/\\')
    return sanitized[:31] or f"Table{index:02d}"


__all__ = [
    "ensure_tableformer_structure",
    "clip_overlapping_table_cells",
    "write_table_metadata",
    "export_tables_to_xlsx",
    "sanitize_table_bboxes",
]
