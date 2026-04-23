# CLAUDE.md

Offline PDF extraction CLI built on IBM's Docling. Converts PDFs to Markdown, JSON, HTML, YAML, text, DocTags, and Excel with advanced table handling and optional OCR repair.

## Commands

```bash
# Setup
make venv && make install          # CUDA (requirements.txt has torch+cu121)
# Or: make venv && pip install -r requirements-cpu.txt  # CPU-only

# Prefetch models (one-time, enables offline use)
make prefetch
# Or: python -m docling_offline prefetch-models --device cuda

# Convert PDFs
make convert INPUTS="docs/*.pdf"
# Or: python -m docling_offline convert docs/sample.pdf --output out --format md json

# Full conversion with OCR repair
python -m docling_offline convert docs/*.pdf \
  --output out --format all \
  --table-mode accurate --no-table-cell-matching \
  --ocr easyocr --device cuda

# Extract tables to CSV from existing Docling JSON
python -m docling_offline extract-tables out/sample.json --output tables

# Dev
make install-dev                   # black, ruff, pytest
make test                          # pytest (stubs Docling, no models needed)
make format                        # black
make lint                          # ruff
```

## Architecture

```
src/docling_offline/
  cli.py          — Typer CLI: prefetch-models, convert, extract-tables
  config.py       — Frozen dataclasses (PrefetchConfig, ConvertConfig), path normalization, PDF discovery
  processor.py    — Core pipeline: builds Docling converter, runs conversion, detects garbled tables, OCR repair
  postprocess.py  — Table extraction to CSV
  utils.py        — Table bbox sanitization, cell clipping, garbled text detection, OCR repair, Excel export
scripts/
  repair_tables.py       — Standalone OCR repair for targeted pages
  download_easyocr_weights.py
bin/
  check_bounds.py        — Table bounding box visualization (matplotlib)
```

## Key Concepts

**Garbled table detection** — Analyzes text quality (printable/alnum ratios, control chars) per table cell to flag OCR failures.

**Two-level OCR repair** — First retries full-page OCR on suspect pages, then falls back to cell-level crop-and-OCR for remaining garbled cells.

**Table cell matching** — Docling's default mode vs `--no-table-cell-matching` with overlap clipping for difficult layouts.

**Offline-first** — `prefetch-models` downloads all artifacts once. Subsequent runs need no network.

## CLI Options (convert)

| Flag | Default | Purpose |
|------|---------|---------|
| `--format` | `md json` | Output formats: md, json, html, text, doctags, yaml, xlsx, all |
| `--device` | `auto` | auto, cuda, mps, cpu |
| `--ocr` | `none` | none, auto, easyocr, tesseract, tesseract-cli, rapidocr |
| `--table-mode` | `fast` | fast or accurate |
| `--workers` | `1` | File-level parallelism (tune up carefully for OOM) |
| `--layout-batch-size` | `4` | Layout model batch size |
| `--export-tables-xlsx` | off | Also emit per-doc Excel table export |
| `--no-table-cell-matching` | off | Disable cell matching, use overlap clipping instead |
| `--force-full-page-ocr` | off | OCR every page, not just garbled ones |

## Output

Each conversion produces per-document files in the output directory:
- `<stem>.{md,json,html,text,yaml,doctags}` — requested formats
- `<stem>.tables.json` — table metadata (bounding boxes, repair info)
- `<stem>.tables.xlsx` — optional Excel export

## Conventions

- Python 3.10–3.12, Black for formatting, Ruff for linting
- Frozen dataclasses with `__post_init__` validation (raises ConfigError)
- Full type hints on public functions
- Per-file ConversionResult objects for error tracking
- Tests stub Docling classes in conftest.py — fast, no model downloads needed
