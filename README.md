# Offline Docling PDF Extraction Tool

`docling-offline` converts PDF collections into Markdown, JSON, HTML, plain text, or DocTags outputs using IBM's [Docling](https://github.com/doclingproject/docling) pipeline. The tool is engineered for offline, repeatable workflows on laptops and workstations—once models are prefetched, conversions run without an internet connection.

JSON is the default export format; specify `--format` flags to add Markdown, HTML, or other outputs.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m docling_offline prefetch-models --artifacts-path .artifacts --device cuda
python -m docling_offline convert docs/Whitestone_page24.pdf --output out
# add extra formats if you need them, e.g.:
# python -m docling_offline convert docs/Whitestone_page24.pdf --output out --format md --format html
```

## Installation

1. Install Python 3.11+ and create a virtualenv.
2. `pip install -r requirements.txt` installs Docling, CUDA-enabled PyTorch, pytest/ruff/black, and OCR extras.
3. Optional: `pip install -e .` if you want the `docling-offline` console script instead of `python -m docling_offline`.

## Usage

1. Connect once, run `docling-offline prefetch-models` to populate `.artifacts/`.
2. Disconnect from the network. All `convert` runs reuse the cached models.
3. Share the `.artifacts/` directory across machines for deterministic builds.

**Examples**

- Convert a single PDF to JSON (default format) with GPU acceleration:
  ```bash
  python -m docling_offline convert docs/Whitestone_page24.pdf --output out --device cuda --artifacts-path .artifacts
  ```
- Request every supported artifact in one go:
  ```bash
  python -m docling_offline convert docs/Whitestone_page24.pdf \
    --output out \
    --format json --format md --format html --format text --format doctags \
    --device cuda --artifacts-path .artifacts
  ```
- Convert a directory recursively:
  ```bash
  python -m docling_offline convert docs --output out --format json --device cuda
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

## Testing

- `python -m pytest` runs the unit suite; stubs replace Docling so tests stay fast and offline.
- Optional: `make format`, `make lint`, and `make test` shortcuts are provided in the `Makefile`.

## Troubleshooting

- **`docling` import errors:** Verify `pip install -r requirements.txt` succeeded and Python ≥3.10 is active.
- **CUDA init failures:** Confirm the installed Torch wheel matches your driver/CUDA toolkit, or fall back to `--device cpu`.
- **Tesseract not found:** Install the system `tesseract` binary and ensure it is on `PATH`, or switch to `easyocr` / `rapidocr`.
- **Out of memory:** Reduce `--workers` and batch sizes, disable OCR, or split the input directory into smaller batches.
