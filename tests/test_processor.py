from pathlib import Path
from typing import Dict, List

import pytest

from docling_offline import processor
from docling_offline.config import ConvertConfig, PrefetchConfig


def test_convert_documents_writes_all_formats(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_text("fake pdf")

    config = ConvertConfig(
        inputs=(pdf,),
        output=tmp_path / "out",
        formats=("md", "json", "html", "text", "doctags"),
        artifacts_path=tmp_path / ".artifacts",
    )

    results = processor.convert_documents(config)
    assert len(results) == 1
    result = results[0]
    assert result.success
    for fmt in config.formats:
        target = config.output / f"{pdf.stem}.{fmt}"
        assert target.exists()


def test_prefetch_models_invokes_docling(tmp_path: Path, stub_docling: Dict[str, object]) -> None:
    processor.prefetch_models(
        PrefetchConfig(
            artifacts_path=tmp_path / ".artifacts",
            device="cpu",
            threads=2,
        )
    )
    standard_pipeline = stub_docling["StandardPdfPipeline"]
    assert getattr(standard_pipeline, "calls", 0) == 1
