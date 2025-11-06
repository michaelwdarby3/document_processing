# Offline Docling PDF Extraction Tool

`docling-offline` converts PDF collections into Markdown, JSON, HTML, plain text, or DocTags outputs using IBM's [Docling](https://github.com/doclingproject/docling) pipeline. The tool is engineered for offline, repeatable workflows on laptops and workstations—once models are prefetched, conversions run without an internet connection.

JSON is the default export format; specify `--format` flags to add Markdown, HTML, or other outputs.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m docling_offline prefetch-models --artifacts-path .artifacts
python -m docling_offline convert ./samples --output out
# add extra formats if you need them, e.g.: --format md --format html
```

## Installation

- **Runtime:** Install base dependencies from `requirements.txt`. This keeps the footprint small for offline deployments.
- **OCR extras:** Run `pip install -r requirements-ocr-extras.txt` to enable optional OCR engines (EasyOCR, Tesseract, RapidOCR).
- **Developer tooling:** `pip install -r requirements-dev.txt` installs pytest, Black, and Ruff for local validation. The bundle includes Docling and its heavier dependencies; expect a large download (PyTorch, OCR models). For lightweight CI or stubbed unit tests, `pip install --no-deps -r requirements-dev.txt` followed by `pip install -r requirements-lite.txt` skips the GPU-sized wheels while keeping the developer toolchain.
- **Lock file:** `requirements.lock.txt` pins the stack. Regenerate with `make lock` after dependency bumps.

## Offline Usage

1. Connect once, run `docling-offline prefetch-models` to populate `.artifacts/`.
2. Disconnect from the network. All `convert` runs reuse the cached models.
3. Share the `.artifacts/` directory across machines for deterministic builds.

## Device Notes

- **Apple Silicon:** `--device auto` finds Metal (MPS). Force with `--device mps` if needed.
- **NVIDIA GPU:** Install a CUDA-compatible PyTorch wheel, then use `--device cuda`.
- **CPU-only:** Leave the default `auto` or specify `--device cpu` for maximum compatibility.

## OCR Guidance

- OCR is off by default (`--ocr none`). Digital PDFs convert faster without OCR passes.
- Enable OCR per engine: `easyocr`, `tesseract`, `tesseract-cli`, `rapidocr`, `ocrmac` (macOS).
- Set languages with `--ocr-langs "en,de"`; use `--force-full-page-ocr` for low-quality scans.

## Tuning & Concurrency

- Start with defaults: `--workers 1` and batch sizes of `4`.
- Increase `--workers` gradually on machines with ≥32 GB RAM or ≥8 GB VRAM.
- Tune `--layout-batch-size`, `--ocr-batch-size`, `--table-batch-size` upward (16–64) when running on powerful GPUs.
- The CLI warns when multiple workers are enabled to prevent accidental OOM on laptops.

## Testing & Tooling

- Run `make lint` (Ruff) and `make format` (Black) before opening a PR.
- Execute `make test` for pytest-based unit tests. They work without downloading Docling models; stubs handle the conversion layer.

## Troubleshooting

- **`docling` import errors:** Verify `pip install -r requirements.txt` succeeded and Python ≥3.10 is active.
- **CUDA init failures:** Confirm the installed Torch wheel matches your driver/CUDA toolkit, or fall back to `--device cpu`.
- **Tesseract not found:** Install the system `tesseract` binary and ensure it is on `PATH`, or switch to `easyocr` / `rapidocr`.
- **Out of memory:** Reduce `--workers` and batch sizes, disable OCR, or split the input directory into smaller batches.
