#!/usr/bin/env python3
"""Force EasyOCR to download its weights into .artifacts/EasyOcr."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EasyOCR weights into the Docling artifacts cache.")
    parser.add_argument(
        "--artifacts-path",
        default=Path(".artifacts"),
        type=Path,
        help="Base artifacts directory (defaults to .artifacts)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en"],
        help="EasyOCR language codes to download (default: en).",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU when initializing EasyOCR (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.artifacts_path.expanduser().resolve() / "EasyOcr"
    target.mkdir(parents=True, exist_ok=True)

    # Lazy import so this script can fail cleanly if easyocr isn't available.
    from easyocr import Reader

    print(f"Downloading EasyOCR weights into {target} for languages {args.langs}")
    Reader(args.langs, gpu=args.gpu, model_storage_directory=str(target), download_enabled=True)
    print("EasyOCR weights ready.")


if __name__ == "__main__":
    main()
