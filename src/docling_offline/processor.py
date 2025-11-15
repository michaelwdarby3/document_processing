"""Core processing helpers for the docling-offline CLI."""

from __future__ import annotations

import functools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set

import yaml

if TYPE_CHECKING:  # pragma: no cover - typing only
    from docling.document_converter import DocumentConverter as _DocumentConverter
else:  # pragma: no cover - runtime alias
    _DocumentConverter = Any

try:  # pragma: no cover - optional dependency
    from markdown import markdown as _render_markdown
except ImportError:  # pragma: no cover
    _render_markdown = None

from .config import ConvertConfig, PrefetchConfig, output_paths_for, safe_stem
from .utils import (
    ensure_tableformer_structure,
    clip_overlapping_table_cells,
    write_table_metadata,
    export_tables_to_xlsx,
    sanitize_table_bboxes,
    detect_garbled_tables,
    merge_repaired_tables,
    _write_text_with_fallback,
)

logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    source: Path
    success: bool
    outputs: List[Path]
    message: str = ""


def prefetch_models(config: PrefetchConfig) -> None:
    """Warm Docling caches so future runs are offline-ready."""

    resources = _load_docling()
    StandardPdfPipeline = resources["StandardPdfPipeline"]
    snapshot_download = resources.get("snapshot_download")
    PathlibPath = resources["Path"]

    download_fn = getattr(StandardPdfPipeline, "download_models_hf", None)
    if callable(download_fn):
        try:
            download_fn(local_dir=config.artifacts_path, force=False)
            return
        except Exception:
            # Fall back to converter-based prefetch below
            pass

    if snapshot_download:
        pipeline_defaults = StandardPdfPipeline.get_default_options()
        layout_options = getattr(pipeline_defaults, "layout_options", None)
        layout_spec = getattr(layout_options, "model_spec", None) if layout_options else None
        artifacts_root = PathlibPath(config.artifacts_path)
        hf_targets = []
        if layout_spec is not None:
            repo_folder = getattr(layout_spec, "model_repo_folder", None)
            repo_id = getattr(layout_spec, "repo_id", None)
            revision = getattr(layout_spec, "revision", "main")
            if repo_id and repo_folder:
                hf_targets.append(
                    (
                        repo_id,
                        revision,
                        artifacts_root / repo_folder,
                    )
                )
        hf_targets.append(
            (
                "ds4sd/docling-models",
                "v2.3.0",
                artifacts_root / "ds4sd--docling-models",
            )
        )
        for repo_id, revision, dest in hf_targets:
            try:
                dest.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(dest),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except Exception:
                continue
        ensure_tableformer_structure(artifacts_root)

    try:
        converter = _build_converter(
            artifacts_path=config.artifacts_path,
            device=config.device,
            threads=config.threads,
            layout_batch_size=4,
            ocr_batch_size=4,
            table_batch_size=4,
            ocr="none",
            ocr_langs=("en",),
            force_full_page_ocr=False,
            generate_page_images=False,
            table_mode="fast",
            table_cell_matching=True,
        )
        converter.convert(Path(__file__))
    except Exception:
        # We only needed the side effect of downloading models.
        pass


def convert_documents(config: ConvertConfig) -> List[ConversionResult]:
    """Convert all documents described by the configuration."""

    tasks = list(config.discovered_inputs)
    if not tasks:
        return []

    if config.workers == 1:
        converter = _build_converter(
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
        return [_convert_single(path, config, converter) for path in tasks]

    worker_count = min(config.workers, multiprocessing.cpu_count())
    worker_fn = functools.partial(_convert_single_worker, config=config)
    results: List[ConversionResult] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(worker_fn, path): path for path in tasks}
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                message = str(exc)
                results.append(
                    ConversionResult(source=path, success=False, outputs=[], message=message)
                )
                if config.fail_fast:
                    for remaining in future_map:
                        remaining.cancel()
                    break
            else:
                results.append(result)
                if config.fail_fast and not result.success:
                    for remaining in future_map:
                        remaining.cancel()
                    break
    return results


def _convert_single_worker(path: Path, config: ConvertConfig) -> ConversionResult:
    converter = _build_converter(
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
    return _convert_single(path, config, converter)


def _convert_single(path: Path, config: ConvertConfig, converter: _DocumentConverter) -> ConversionResult:
    outputs = output_paths_for(config.formats, config.output, path)
    try:
        resources = _load_docling()
        ConversionStatus = resources["ConversionStatus"]
        result = converter.convert(path)
        success_statuses = {getattr(ConversionStatus, "SUCCESS", None), getattr(ConversionStatus, "SUCCEEDED", None), getattr(ConversionStatus, "PARTIAL_SUCCESS", None)}
        if result.status not in {status for status in success_statuses if status is not None}:
            return ConversionResult(
                source=path,
                success=False,
                outputs=[],
                message=f"conversion status {result.status.name.lower()}",
            )
        document = result.document
        suspect_map, suspect_indexes = detect_garbled_tables(document)
        repaired_tables: Set[int] = set()
        page_metadata = sanitize_table_bboxes(document)
        if suspect_map:
            logger.warning(
                "Table text looked garbled on pages: %s",
                ", ".join(str(p) for p in sorted(suspect_map.keys())),
            )
            repaired_tables = _repair_tables_with_ocr(path, config, suspect_map, document)
            if repaired_tables:
                page_metadata = sanitize_table_bboxes(document)
            fallback = _fallback_cell_ocr(path, config, suspect_map, document)
            if fallback:
                repaired_tables.update(fallback)
                page_metadata = sanitize_table_bboxes(document)
        clipped_cells_total = 0
        clipped_by_table: Dict[int, int] = {}
        if not config.table_cell_matching and config.clip_table_overlap:
            clipped_cells_total, clipped_by_table = clip_overlapping_table_cells(document)
            if clipped_cells_total:
                page_metadata = sanitize_table_bboxes(document)
            if clipped_cells_total:
                logger.info("Clipped %d cells due to overlapping tables", clipped_cells_total)
        if not suspect_map:
            logger.info("No garbled tables detected")
        written = _export_document(document, config.formats, outputs)
        metadata_path = config.output / f"{safe_stem(path)}.tables.json"
        write_table_metadata(
            document,
            metadata_path,
            clipping_enabled=(not config.table_cell_matching and config.clip_table_overlap),
            clipped_cells_total=clipped_cells_total,
            clipped_by_table=clipped_by_table,
            table_mode=config.table_mode,
            table_cell_matching=config.table_cell_matching,
            formats=config.formats,
            page_metadata=page_metadata,
            garbled_tables=suspect_indexes,
            repaired_tables=repaired_tables,
        )
        written.append(metadata_path)
        if config.export_tables_xlsx and "xlsx" not in config.formats:
            xlsx_path = config.output / f"{safe_stem(path)}.tables.xlsx"
            workbook = export_tables_to_xlsx(document, xlsx_path)
            if workbook:
                written.append(workbook)
        return ConversionResult(source=path, success=True, outputs=written)
    except Exception as exc:
        return ConversionResult(
            source=path,
            success=False,
            outputs=[],
            message=str(exc),
        )


def _export_document(document: object, formats: Sequence[str], outputs: Sequence[Path]) -> List[Path]:
    written: List[Path] = []
    for fmt, target in zip(formats, outputs):
        target.parent.mkdir(parents=True, exist_ok=True)
        created_path: Optional[Path] = None
        if fmt == "md":
            content = document.export_to_markdown()
            created_path = _write_text_with_fallback(Path(target), content, encoding="utf-8")
        elif fmt == "text":
            content = document.export_to_text()
            created_path = _write_text_with_fallback(Path(target), content, encoding="utf-8")
        elif fmt == "json":
            content = document.export_to_dict()
            payload = json.dumps(content, ensure_ascii=False, indent=2)
            created_path = _write_text_with_fallback(Path(target), payload, encoding="utf-8")
        elif fmt == "html":
            html_method = getattr(document, "export_to_html", None)
            if callable(html_method):
                html_output = html_method()
            else:
                markdown_text = document.export_to_markdown()
                if _render_markdown:
                    html_output = _render_markdown(markdown_text)
                else:
                    html_output = markdown_text.replace("\n", "<br/>")
            created_path = _write_text_with_fallback(Path(target), html_output, encoding="utf-8")
        elif fmt == "doctags":
            export_fn = getattr(document, "export_to_doctags", None)
            if not callable(export_fn):
                export_fn = getattr(document, "export_to_document_tokens", None)
            if not callable(export_fn):
                raise RuntimeError("Current docling build does not support doctags export.")
            content = export_fn()
            created_path = _write_text_with_fallback(Path(target), content, encoding="utf-8")
        elif fmt == "yaml":
            data = document.export_to_dict()
            payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
            created_path = _write_text_with_fallback(Path(target), payload, encoding="utf-8")
        elif fmt == "xlsx":
            workbook_path = export_tables_to_xlsx(document, Path(target))
            if workbook_path:
                created_path = workbook_path
        else:
            raise RuntimeError(f"Format '{fmt}' is not supported by the installed docling version.")
        if created_path:
            written.append(created_path)
    return written


def _build_converter(
    *,
    artifacts_path: Path,
    device: str,
    threads: int,
    layout_batch_size: int,
    ocr_batch_size: int,
    table_batch_size: int,
    ocr: str,
    ocr_langs: Sequence[str],
    force_full_page_ocr: bool,
    generate_page_images: bool,
    table_mode: str,
    table_cell_matching: bool,
) -> _DocumentConverter:
    resources = _load_docling()
    DocumentConverter = resources["DocumentConverter"]
    FormatOption = resources["FormatOption"]
    InputFormat = resources["InputFormat"]
    StandardPdfPipeline = resources["StandardPdfPipeline"]
    EasyOcrOptions = resources["EasyOcrOptions"]
    TesseractCliOcrOptions = resources["TesseractCliOcrOptions"]
    TesseractOcrOptions = resources["TesseractOcrOptions"]
    PdfPipelineOptions = resources["PdfPipelineOptions"]
    PdfBackend = resources["PdfBackend"]

    ocr_engine = ocr.lower()
    if ocr_engine == "none":
        ocr_options = EasyOcrOptions(use_gpu=device == "cuda", lang=list(ocr_langs))
        do_ocr = False
    elif ocr_engine in {"auto", "easyocr"}:
        ocr_options = EasyOcrOptions(use_gpu=device == "cuda", lang=list(ocr_langs))
        do_ocr = True
    elif ocr_engine == "tesseract-cli":
        ocr_options = TesseractCliOcrOptions(lang=list(ocr_langs))
        do_ocr = True
    elif ocr_engine == "tesseract":
        ocr_options = TesseractOcrOptions(lang=list(ocr_langs))
        do_ocr = True
    else:
        raise RuntimeError(f"OCR mode '{ocr}' is not supported by the installed docling version.")

    pipeline_defaults = StandardPdfPipeline.get_default_options()
    pipeline_update = {
        "artifacts_path": artifacts_path,
        "do_ocr": do_ocr,
        "ocr_options": ocr_options,
        "generate_page_images": generate_page_images,
    }
    # Copy across legacy CLI knobs if Docling exposes them on the options model.
    possible_fields = (
        ("device", device),
        ("threads", threads),
        ("layout_batch_size", layout_batch_size),
        ("ocr_batch_size", ocr_batch_size),
        ("table_batch_size", table_batch_size),
        ("force_full_page_ocr", force_full_page_ocr),
    )
    for field, value in possible_fields:
        if hasattr(pipeline_defaults, field):
            pipeline_update[field] = value

    TableFormerMode = resources.get("TableFormerMode")
    TableStructureOptions = resources.get("TableStructureOptions")
    if TableFormerMode and TableStructureOptions:
        table_mode_enum = TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
        base_table_options = getattr(pipeline_defaults, "table_structure_options", None)
        if base_table_options is None:
            base_table_options = TableStructureOptions()
        table_options = base_table_options.model_copy(
            update={"mode": table_mode_enum, "do_cell_matching": table_cell_matching}
        )
        pipeline_update["table_structure_options"] = table_options
    elif table_mode != "fast" or not table_cell_matching:
        print(
            "[docling-offline] WARNING: table-mode/table-cell-matching requested, "
            "but the installed Docling package does not expose TableFormer APIs. "
            "These flags will be ignored."
        )

    pipeline_options = pipeline_defaults.model_copy(update=pipeline_update)

    format_options = {
        InputFormat.PDF: FormatOption(
            backend=PdfBackend,
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }

    return DocumentConverter(format_options=format_options)


def _repair_tables_with_ocr(pdf_path: Path, config: ConvertConfig, suspect_map: Dict[int, List[int]], document: Any) -> Set[int]:
    if not suspect_map:
        return set()
    repair_converter = _build_converter(
        artifacts_path=config.artifacts_path,
        device=config.device,
        threads=config.threads,
        layout_batch_size=config.layout_batch_size,
        ocr_batch_size=config.ocr_batch_size,
        table_batch_size=config.table_batch_size,
        ocr="easyocr",
        ocr_langs=config.ocr_langs,
        force_full_page_ocr=True,
        generate_page_images=config.generate_page_images,
        table_mode=config.table_mode,
        table_cell_matching=config.table_cell_matching,
    )
    repaired_tables: Set[int] = set()
    weights_missing = False
    for page_no, table_indexes in suspect_map.items():
        logger.info("Running OCR repair on page %d for tables %s", page_no, table_indexes)
        try:
            result = repair_converter.convert(pdf_path, page_range=(page_no, page_no))
        except FileNotFoundError as exc:
            logger.error("OCR repair failed on page %d: %s", page_no, exc)
            logger.error(
                "EasyOCR weights are missing. Enable downloads once (network required) via "
                "`python -m docling_offline prefetch-models --device %s --artifacts-path %s` "
                "or run EasyOCR manually to populate %s.",
                config.device,
                config.artifacts_path,
                exc.filename or "the EasyOCR cache",
            )
            weights_missing = True
            break
        except Exception as exc:
            logger.error("OCR repair failed on page %d: %s", page_no, exc)
            continue
        repair_doc = result.document
        sanitize_table_bboxes(repair_doc)
        replaced = merge_repaired_tables(document, repair_doc, table_indexes, page_no)
        repaired_tables.update(replaced)
        if replaced:
            logger.info("Replaced %d tables on page %d via OCR", len(replaced), page_no)
        else:
            logger.warning("OCR repair on page %d produced no matching tables", page_no)
    if weights_missing and len(repaired_tables) < len(suspect_map):
        logger.warning("Skipping remaining OCR repairs because EasyOCR weights are unavailable.")
    return repaired_tables


def _fallback_cell_ocr(
    pdf_path: Path,
    config: ConvertConfig,
    tables_by_page: Dict[int, List[int]],
    document: Any,
) -> Set[int]:
    if not tables_by_page:
        return set()
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover
        logger.error("pypdfium2 is required for cell-level OCR repairs but is not installed.")
        return set()
    try:
        from easyocr import Reader
    except ImportError:  # pragma: no cover
        logger.error("easyocr is required for cell-level OCR repairs but is not installed.")
        return set()

    gpu = config.device.startswith("cuda")
    reader = Reader(
        list(config.ocr_langs),
        gpu=gpu,
        model_storage_directory=str(config.artifacts_path / "EasyOcr"),
        download_enabled=False,
    )
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    repaired: Set[int] = set()
    scale = 4.0

    for page_no, indexes in tables_by_page.items():
        logger.info("Running fallback cell OCR on page %d for tables %s", page_no, indexes)
        try:
            page = pdf_doc[page_no - 1]
        except IndexError:
            logger.error("Page %d not found in PDF; skipping cell-level OCR.", page_no)
            continue
        renderer = page.render(scale=scale)
        pil_image = renderer.to_pil()
        page_width = page.get_width()
        page_height = page.get_height()
        changed_tables = 0
        for table_index in indexes:
            try:
                table = document.tables[table_index]
            except IndexError:
                continue
            data = getattr(table, "data", None)
            if data is None:
                continue
            changed = False
            for cell in getattr(data, "table_cells", []) or []:
                bbox = getattr(cell, "bbox", None)
                if bbox is None:
                    continue
                top_bbox = (
                    bbox
                    if getattr(bbox, "coord_origin", None) == getattr(bbox, "coord_origin", None).__class__.TOPLEFT
                    else bbox.to_top_left_origin(page_height)
                )
                x0 = max(int(top_bbox.l * scale), 0)
                y0 = max(int(top_bbox.t * scale), 0)
                x1 = min(int(top_bbox.r * scale), pil_image.width)
                y1 = min(int(top_bbox.b * scale), pil_image.height)
                if x1 - x0 < 6 or y1 - y0 < 6:
                    continue
                crop = pil_image.crop((x0, y0, x1, y1))
                crop = crop.convert("L")
                crop_arr = np.asarray(crop)
                if crop_arr.size == 0:
                    continue
                results = reader.readtext(crop_arr, detail=0, paragraph=True)
                ocr_text = " ".join(results).strip()
                if _looks_reasonable(ocr_text) and ocr_text != getattr(cell, "text", ""):
                    cell.text = ocr_text
                    changed = True
            if changed:
                repaired.add(table_index)
                changed_tables += 1
        if changed_tables:
            logger.info("Rewrote %d tables on page %d via fallback cell OCR.", changed_tables, page_no)
        else:
            logger.warning("Fallback cell OCR on page %d did not improve any tables.", page_no)
    return repaired


def _looks_reasonable(text: str) -> bool:
    if not text:
        return False
    printable = sum(1 for ch in text if ch.isprintable())
    alnum = sum(1 for ch in text if ch.isalnum())
    return printable / len(text) > 0.5 and alnum / len(text) > 0.2


def summarize_results(results: Sequence[ConversionResult]) -> Dict[str, int]:
    stats = {"succeeded": 0, "failed": 0}
    for result in results:
        if result.success:
            stats["succeeded"] += 1
        else:
            stats["failed"] += 1
    return stats


@lru_cache(maxsize=1)
def _load_docling() -> Dict[str, Any]:
    try:
        from docling.document_converter import (
            DocumentConverter,
            FormatOption,
            InputFormat,
            ConversionStatus,
        )  # type: ignore
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
            TesseractCliOcrOptions,
            TesseractOcrOptions,
        )
        try:  # Optional across docling releases
            from docling.datamodel.pipeline_options import TableStructureOptions  # type: ignore
        except ImportError:  # pragma: no cover
            TableStructureOptions = None  # type: ignore
        try:
            from docling.datamodel.pipeline_options import TableFormerMode as _PipelineTableFormerMode  # type: ignore
        except ImportError:  # pragma: no cover
            _PipelineTableFormerMode = None  # type: ignore
        if _PipelineTableFormerMode is not None:
            TableFormerMode = _PipelineTableFormerMode
        else:  # pragma: no cover - legacy fallback
            try:
                from docling.models.tableformer.tableformer_config import TableFormerMode  # type: ignore
            except ImportError:
                TableFormerMode = None  # type: ignore
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "docling is not installed or incompatible. Install dependencies via `pip install -r requirements.txt`."
        ) from exc

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None

    return {
        "DocumentConverter": DocumentConverter,
        "FormatOption": FormatOption,
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
        "PdfBackend": PyPdfiumDocumentBackend,
        "StandardPdfPipeline": StandardPdfPipeline,
        "EasyOcrOptions": EasyOcrOptions,
        "TesseractCliOcrOptions": TesseractCliOcrOptions,
        "TesseractOcrOptions": TesseractOcrOptions,
        "TableStructureOptions": TableStructureOptions,
        "TableFormerMode": TableFormerMode,
        "ConversionStatus": ConversionStatus,
        "snapshot_download": snapshot_download,
        "Path": Path,
    }
