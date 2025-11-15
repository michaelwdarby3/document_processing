# Offline Docling PDF Extraction Tool

`docling-offline` converts PDF collections into Markdown, JSON, HTML, YAML, plain text, DocTags, or Excel workbooks using IBM's [Docling](https://github.com/doclingproject/docling) pipeline. The tool is engineered for offline, repeatable workflows on laptops and workstations—once models are prefetched, conversions run without an internet connection.

JSON is the default export format; specify `--format` flags to add Markdown, HTML, YAML, Excel (`--format xlsx`), or other outputs. Use `--format all` to emit every supported artifact (including the Excel workbook) in a single run.

Whenever Docling encounters a table whose text stream looks mangled (e.g., custom-encoded glyphs), the CLI automatically re-runs EasyOCR on that page and splices the OCR’d table back into the outputs. The metadata file (`*.tables.json`) records which tables were flagged/repaired.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m docling_offline prefetch-models --artifacts-path .artifacts --device cuda
python -m docling_offline convert docs/sample.pdf --output out --device cuda
# add extra formats if you need them, e.g.:
# python -m docling_offline convert docs/sample.pdf --output out --format md --format html
```

## Installation

1. Install Python 3.11+ and create a virtualenv.
2. `pip install -r requirements.txt` installs Docling, optimized PyTorch (GPU), pytest/ruff/black, and OCR extras.
3. Optional: `pip install -e .` if you want the `docling-offline` console script instead of `python -m docling_offline`.

### CPU-only setup

If you want a strictly CPU environment (no CUDA dependencies), create a parallel venv and install from `requirements-cpu.txt`:

```bash
python3 -m venv .venv-cpu && source .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m docling_offline convert docs/sample.pdf --output out --device cpu --artifacts-path .artifacts
```

## Usage

1. Connect once, run `docling-offline prefetch-models` to populate `.artifacts/`.
2. Disconnect from the network. All `convert` runs reuse the cached models.
3. Share the `.artifacts/` directory across machines for deterministic builds.

**Examples**

- Convert a single PDF to JSON (default format) with GPU acceleration:
  ```bash
  python -m docling_offline convert docs/sample.pdf --output out --device cuda --artifacts-path .artifacts
  ```
- Request every supported artifact (JSON, Markdown, HTML, YAML, text, DocTags, Excel) in one go:
  ```bash
  python -m docling_offline convert docs/sample.pdf \
    --output out --format all \
    --device cuda --artifacts-path .artifacts
  ```
- Emit only the Excel workbook (each table becomes its own sheet):
  ```bash
  python -m docling_offline convert docs/sample.pdf \
    --output out --format xlsx \
    --device cuda --artifacts-path .artifacts
  ```
- Try the more accurate table model (slower but may split columns better):
  ```bash
  python -m docling_offline convert docs/sample.pdf \
    --output out --format json \
    --table-mode accurate --table-cell-matching \
    --device cuda --artifacts-path .artifacts
  ```
- Run the accurate TableFormer without cell matching (keeps raw columns), trim overlaps, and emit every format (JSON/MD/HTML/Text/YAML/DocTags/XLSX) plus metadata:
  ```bash
  python -m docling_offline convert docs/sample.pdf \
    --output out_no_match \
    --format all \
    --table-mode accurate --no-table-cell-matching \
    --device cuda --artifacts-path .artifacts
  ```
- Same as above but force-emit the Excel workbook even if you’re only exporting JSON (useful for debugging):
  ```bash
  python -m docling_offline convert docs/sample.pdf \
    --output out_no_match \
    --format json \
    --table-mode accurate --no-table-cell-matching \
    --export-tables-xlsx \
    --device cuda --artifacts-path .artifacts
  ```
- Convert a long PDF to every format with overlap clipping and metadata (replace with your file path):
  ```bash
  python -m docling_offline convert docs/Whitestone_Facility_Maintenance_And_Repair_Cost_Reference_2009-2010.pdf \
    --output out_nomatch \
    --format all \
    --table-mode accurate --no-table-cell-matching \
    --device cuda --artifacts-path .artifacts
  ```
- CPU-only run (useful on machines without CUDA/MPS; pair with `requirements-cpu.txt`):
  ```bash
  python -m docling_offline convert docs/sample.pdf \
    --output cpu_out --format json \
    --table-mode accurate --table-cell-matching \
    --device cpu --artifacts-path .artifacts
  ```
- Re-run only the tables that look garbled (automated OCR repair):
  ```bash
  ./scripts/repair_tables.py docs/Whitestone_Facility_Maintenance_And_Repair_Cost_Reference_2009-2010.pdf \
    repaired_out --format json xlsx
  ```
  This wrapper runs the normal conversion, detects tables with noisy encodings, reprocesses just those pages with EasyOCR, and emits repaired JSON/XLSX/metadata.
- Convert a directory recursively:
  ```bash
  python -m docling_offline convert docs --output out --format json --device cuda
  ```
- Export the tables from a Docling JSON into CSV:
  ```bash
  python -m docling_offline extract-tables cuda_out/sample.json --output tables
  ```
- Every conversion also emits `<name>.tables.json`, summarizing each table’s page, rows/columns, bounding box coverage, clipping stats, and which formats were requested; combine this with `--format xlsx` to inspect tables quickly in Excel.
- Visualize table bounding boxes:
  ```bash
  python bin/check_bounds.py docs/sample.pdf cuda_out/sample.json --table 0 --page 0
  ```

Each run prints an elapsed time along with the success/failure summary.

## Device Notes

- **Apple Silicon:** `--device auto` finds Metal (MPS). Force with `--device mps` if needed.
- **NVIDIA GPU:** Install a CUDA-compatible PyTorch wheel, then use `--device cuda`.
- **CPU-only:** Leave the default `auto` or specify `--device cpu` for maximum compatibility.

## OCR Setup

- OCR is off by default (`--ocr none`). Digital PDFs convert faster without OCR passes.
- Enable OCR per engine: `easyocr`, `tesseract`, `tesseract-cli`, `rapidocr`, `ocrmac` (macOS).
- Set languages with `--ocr-langs "en,de"`; use `--force-full-page-ocr` for low-quality scans.
- For Tesseract, install the system binary (Ubuntu example: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`).
- Prefetch once with OCR enabled so the weights land in `.artifacts`:
  ```bash
  python -m docling_offline prefetch-models --artifacts-path .artifacts --device cuda
  ```

## Tuning & Concurrency

- Start with defaults: `--workers 1` and batch sizes of `4`.
- Increase `--workers` gradually on machines with ≥32 GB RAM or ≥8 GB VRAM.
- Tune `--layout-batch-size`, `--ocr-batch-size`, `--table-batch-size` upward (16–64) when running on powerful GPUs.
- The CLI warns when multiple workers are enabled to prevent accidental OOM on laptops.
- Table extraction knobs:
  - `--table-mode accurate` enables the heavier TableFormer to try to split columns more aggressively.
  - `--no-table-cell-matching` disables Docling’s heuristic cell merging if it over-merges your layouts.

## Testing

- `python -m pytest` runs the unit suite; stubs replace Docling so tests stay fast and offline.
- Optional: `make format`, `make lint`, and `make test` shortcuts are provided in the `Makefile`.

## Troubleshooting

- **`docling` import errors:** Verify `pip install -r requirements.txt` succeeded and Python ≥3.10 is active.
- **CUDA init failures:** Confirm the installed Torch wheel matches your driver/CUDA toolkit, or fall back to `--device cpu`.
- **Tesseract not found:** Install the system `tesseract` binary and ensure it is on `PATH`, or switch to `easyocr` / `rapidocr`.
- **Out of memory:** Reduce `--workers` and batch sizes, disable OCR, or split the input directory into smaller batches.
