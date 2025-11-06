"""Core processing helpers for the docling-offline CLI."""

from __future__ import annotations

import functools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from docling.document_converter import DocumentConverter as _DocumentConverter
else:  # pragma: no cover - runtime alias
    _DocumentConverter = Any

try:  # pragma: no cover - optional dependency
    from markdown import markdown as _render_markdown
except ImportError:  # pragma: no cover
    _render_markdown = None

from .config import ConvertConfig, PrefetchConfig, output_paths_for

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

    StandardPdfPipeline.download_models_hf(
        local_dir=config.artifacts_path,
        force=False,
    )


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
        written = _export_document(document, config.formats, outputs)
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
        if fmt == "md":
            content = document.export_to_markdown()
            Path(target).write_text(content, encoding="utf-8")
        elif fmt == "text":
            content = document.export_to_text()
            Path(target).write_text(content, encoding="utf-8")
        elif fmt == "json":
            content = document.export_to_dict()
            Path(target).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        elif fmt == "html":
            markdown_text = document.export_to_markdown()
            if _render_markdown:
                html_output = _render_markdown(markdown_text)
            else:  # fallback
                html_output = f"<pre>{escape(markdown_text)}</pre>"
            Path(target).write_text(html_output, encoding="utf-8")
        elif fmt == "doctags":
            content = document.export_to_document_tokens()
            Path(target).write_text(content, encoding="utf-8")
        else:
            raise RuntimeError(f"Format '{fmt}' is not supported by the installed docling version.")
        written.append(target)
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
    pipeline_options = pipeline_defaults.model_copy(
        update={
            "artifacts_path": artifacts_path,
            "do_ocr": do_ocr,
            "ocr_options": ocr_options,
            "generate_page_images": generate_page_images,
        }
    )

    format_options = {
        InputFormat.PDF: FormatOption(
            backend=PdfBackend,
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }

    return DocumentConverter(format_options=format_options)


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
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "docling is not installed or incompatible. Install dependencies via `pip install -r requirements.txt`."
        ) from exc

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
        "ConversionStatus": ConversionStatus,
    }
