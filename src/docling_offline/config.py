"""Configuration helpers and filesystem utilities for docling-offline."""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

SUPPORTED_FORMATS: Set[str] = {"md", "json", "html", "text", "doctags", "yaml", "xlsx"}
ALL_FORMATS: Tuple[str, ...] = tuple(sorted(SUPPORTED_FORMATS))
SUPPORTED_DEVICES: Set[str] = {"auto", "cuda", "mps", "cpu"}
SUPPORTED_OCR_ENGINES: Set[str] = {
    "none",
    "auto",
    "easyocr",
    "tesseract",
    "tesseract-cli",
    "rapidocr",
    "ocrmac",
}
PDF_SUFFIXES: Tuple[str, ...] = (".pdf",)


class ConfigError(ValueError):
    """Raised when CLI or configuration inputs are invalid."""


@dataclass(frozen=True)
class PrefetchConfig:
    """Options required to prime Docling artifacts for offline usage."""

    artifacts_path: Path
    device: str = "auto"
    threads: int = 4

    def __post_init__(self) -> None:
        artifacts = self.artifacts_path.expanduser().resolve()
        object.__setattr__(self, "artifacts_path", artifacts)
        if self.device not in SUPPORTED_DEVICES:
            raise ConfigError(f"Unsupported device '{self.device}'.")
        if self.threads < 1:
            raise ConfigError("threads must be >= 1.")


@dataclass(frozen=True)
class ConvertConfig:
    """Normalized configuration used while converting documents."""

    inputs: Tuple[Path, ...]
    output: Path
    formats: Tuple[str, ...]
    artifacts_path: Path
    device: str = "auto"
    threads: int = 4
    workers: int = 1
    layout_batch_size: int = 4
    ocr_batch_size: int = 4
    table_batch_size: int = 4
    ocr: str = "none"
    ocr_langs: Tuple[str, ...] = ("en",)
    force_full_page_ocr: bool = False
    generate_page_images: bool = False
    fail_fast: bool = False
    table_mode: str = "fast"
    table_cell_matching: bool = True
    clip_table_overlap: bool = False
    export_tables_xlsx: bool = False
    discovered_inputs: Tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized_inputs = tuple(sorted(_normalize_path(p) for p in self.inputs))
        object.__setattr__(self, "inputs", normalized_inputs)

        output_path = _normalize_path(self.output)
        object.__setattr__(self, "output", output_path)

        raw_formats = tuple(fmt.strip() for fmt in self.formats if fmt and fmt.strip())
        if not raw_formats:
            raise ConfigError("At least one output format must be supplied.")

        if any(fmt.lower() == "all" for fmt in raw_formats):
            normalized_formats = ALL_FORMATS
        else:
            normalized_formats = tuple(_normalize_format(fmt) for fmt in raw_formats)
        if not normalized_formats:
            raise ConfigError("At least one output format must be supplied.")
        object.__setattr__(self, "formats", normalized_formats)

        artifacts = _normalize_path(self.artifacts_path)
        object.__setattr__(self, "artifacts_path", artifacts)

        if self.device not in SUPPORTED_DEVICES:
            raise ConfigError(f"Unsupported device '{self.device}'.")
        if self.threads < 1:
            raise ConfigError("threads must be >= 1.")
        if self.workers < 1:
            raise ConfigError("workers must be >= 1.")
        if self.layout_batch_size < 1 or self.ocr_batch_size < 1 or self.table_batch_size < 1:
            raise ConfigError("batch sizes must be >= 1.")

        ocr_engine = self.ocr.lower()
        if ocr_engine not in SUPPORTED_OCR_ENGINES:
            raise ConfigError(f"Unsupported OCR mode '{self.ocr}'.")
        object.__setattr__(self, "ocr", ocr_engine)

        langs = tuple(lang.strip() for lang in self.ocr_langs if lang.strip())
        if not langs:
            raise ConfigError("At least one OCR language is required.")
        object.__setattr__(self, "ocr_langs", langs)

        table_mode_normalized = self.table_mode.strip().lower()
        if table_mode_normalized not in {"fast", "accurate"}:
            raise ConfigError("table_mode must be one of: fast, accurate.")
        object.__setattr__(self, "table_mode", table_mode_normalized)

        discovered = tuple(sorted(_discover_inputs(normalized_inputs)))
        if not discovered:
            raise ConfigError("No PDF inputs were discovered.")
        object.__setattr__(self, "discovered_inputs", discovered)

        # Ensure output directory exists when config is created.
        output_path.mkdir(parents=True, exist_ok=True)


def _normalize_path(value: os.PathLike[str] | str) -> Path:
    path = Path(value).expanduser()
    return path.resolve()


def _normalize_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ConfigError(f"Unsupported format '{value}'. Valid options: {sorted(SUPPORTED_FORMATS)}")
    return fmt


def _discover_inputs(inputs: Iterable[Path]) -> Iterable[Path]:
    seen: Set[Path] = set()
    for item in inputs:
        if item.is_symlink():
            continue
        if item.is_file():
            if item.suffix.lower() in PDF_SUFFIXES:
                normalized = item.resolve()
                if normalized not in seen:
                    seen.add(normalized)
                    yield normalized
            continue
        if item.is_dir():
            for path in item.rglob("*"):
                if path.is_symlink():
                    continue
                if path.is_file() and path.suffix.lower() in PDF_SUFFIXES:
                    normalized = path.resolve()
                    if normalized not in seen:
                        seen.add(normalized)
                        yield normalized


SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_stem(path: Path) -> str:
    """Create a filesystem-safe stem for output artefacts."""

    stem = path.stem
    normalized = SAFE_STEM_PATTERN.sub("_", stem)
    normalized = normalized.strip("._")
    return normalized or "document"


def output_paths_for(formats: Sequence[str], output_dir: Path, source: Path) -> List[Path]:
    stem = safe_stem(source)
    return [output_dir / f"{stem}.{fmt}" for fmt in formats]


def format_list(formats: Sequence[str]) -> str:
    return ", ".join(formats)
