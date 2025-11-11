## Follow-up Tasks

- [ ] Containerize the tool once you need production deployment. Start from a base like `python:3.11-slim`, install the repo’s `requirements.txt` (or `requirements-cpu.txt` for a CPU-only image), add GPU extras (CUDA runtime / `libgl1`, etc.) if needed, install system OCR binaries, and bake in the `.artifacts/` cache for true offline runs.
- [ ] Enable OCR when you encounter scanned PDFs:  
  ```bash
  sudo apt-get install tesseract-ocr  # only if you want Tesseract
  sudo .venv/bin/python -m docling_offline.cli prefetch-models --artifacts-path .artifacts --device cuda
  sudo .venv/bin/python -m docling_offline.cli convert docs --output out --format json --ocr easyocr --ocr-langs en --device cuda --artifacts-path .artifacts
  ```
  EasyOCR is the simplest path—Tesseract works too, but requires the binary and language packs. Prefetching once drops all OCR models into `.artifacts/`.
- [ ] Inspect the upstream `docling` repo (now at `../docling/docling`) for any container examples we might want to borrow before authoring our own Dockerfile.
- [ ] Keep `docs/Whitestone_page24.pdf` handy as the quick sanity-check fixture. It’s a single-page slice of the larger report and converts in ~40 s on CPU; perfect for “does the pipeline still run?” tests.
