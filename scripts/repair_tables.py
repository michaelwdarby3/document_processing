#!/usr/bin/env python3
"""Targeted OCR repair script for Docling tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

from docling_offline.config import ConfigError, ConvertConfig
from docling_offline.processor import _build_converter
from docling_offline.utils import (
    clip_overlapping_table_cells,
    detect_garbled_tables,
    export_tables_to_xlsx,
    merge_repaired_tables,
    sanitize_table_bboxes,
    write_table_metadata,
    _write_text_with_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair garbled tables with targeted OCR re-runs.")
    parser.add_argument("pdf", type=Path, help="Input PDF")
    parser.add_argument("output", type=Path, help="Output directory for repaired artifacts")
    parser.add_argument("--artifacts-path", default=Path(".artifacts"), type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--formats", nargs="*", default=["json", "xlsx", "tables.json"])
    parser.add_argument("--table-mode", default="accurate")
    parser.add_argument("--table-cell-matching", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def build_converter(config: ConvertConfig) -> object:
    return _build_converter(
        artifacts_path=config.artifacts_path,
        device=config.device,
        threads=config.threads,
        layout_batch_size=config.layout_batch_size,
        ocr_batch_size=config.ocr_batch_size,
        table_batch_size=config.table_batch_size,
        ocr=config.ocr,
        ocr_langs=config.ocr_langs,
        force_full_page_ocr=config.force_full_page_ocr,
        generate_page_images=config.generate_page_images,
        table_mode=config.table_mode,
        table_cell_matching=config.table_cell_matching,
    )


def main() -> None:
    args = parse_args()
    pdf = args.pdf.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_config = ConvertConfig(
            inputs=(pdf,),
            output=out_dir,
            formats=tuple(args.formats),
            artifacts_path=args.artifacts_path,
            device=args.device,
            table_mode=args.table_mode,
            table_cell_matching=args.table_cell_matching,
        )
    except ConfigError as exc:  # pragma: no cover
        raise SystemExit(f"configuration error: {exc}") from exc

    base_converter = build_converter(base_config)
    base_result = base_converter.convert(pdf)
    document = base_result.document
    page_metadata = sanitize_table_bboxes(document)
    suspect_map, suspect_indexes = detect_garbled_tables(document)

    print(f"Detected {len(suspect_indexes)} suspect tables across {len(suspect_map)} pages")
    repaired_tables: Set[int] = set()
    if suspect_map:
        repair_config = ConvertConfig(
            inputs=(pdf,),
            output=out_dir,
            formats=("json",),
            artifacts_path=args.artifacts_path,
            device=args.device,
            table_mode=args.table_mode,
            table_cell_matching=args.table_cell_matching,
            ocr="easyocr",
            force_full_page_ocr=True,
        )
        repair_converter = build_converter(repair_config)
        for page_no, table_indexes in suspect_map.items():
            result = repair_converter.convert(pdf, page_range=(page_no, page_no))
            repair_doc = result.document
            sanitize_table_bboxes(repair_doc)
            replaced = merge_repaired_tables(document, repair_doc, table_indexes, page_no)
            repaired_tables.update(replaced)
        if repaired_tables:
            print(f"Repaired {len(repaired_tables)} tables via OCR")
        else:
            print("No OCR repairs were applied (page-level run returned no matches)")
    else:
        print("No garbled tables detected")

    clipped_total, clipped_map = clip_overlapping_table_cells(document)
    if clipped_total:
        print(f"Clipped {clipped_total} overlapping cells")
    else:
        print("No overlap clipping required")

    if "json" in args.formats:
        json_path = out_dir / f"{pdf.stem}.json"
        _write_text_with_fallback(json_path, json.dumps(document.export_to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if "xlsx" in args.formats:
        export_tables_to_xlsx(document, out_dir / f"{pdf.stem}.xlsx")

    metadata_path = out_dir / f"{pdf.stem}.tables.json"
    write_table_metadata(
        document,
        metadata_path,
        clipping_enabled=True,
        clipped_cells_total=clipped_total,
        clipped_by_table=clipped_map,
        table_mode=args.table_mode,
        table_cell_matching=args.table_cell_matching,
        formats=args.formats,
        page_metadata=page_metadata,
        garbled_tables=suspect_indexes,
        repaired_tables=repaired_tables,
    )


if __name__ == "__main__":
    main()
