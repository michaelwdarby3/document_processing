from pathlib import Path

import pytest

from docling_offline.config import ALL_FORMATS, ConfigError, ConvertConfig, PrefetchConfig, safe_stem


def test_convert_config_discovers_pdfs(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    nested = tmp_path / "docs" / "nested"
    nested.mkdir()
    pdf_a = tmp_path / "alpha.pdf"
    pdf_a.write_bytes(b"%PDF-1.7 stub")
    pdf_b = nested / "beta.PDF"
    pdf_b.write_bytes(b"%PDF-1.7 stub")

    config = ConvertConfig(
        inputs=(tmp_path, tmp_path / "docs"),
        output=tmp_path / "out",
        formats=("md", "json"),
        artifacts_path=tmp_path / ".artifacts",
    )

    assert set(config.discovered_inputs) == {pdf_a.resolve(), pdf_b.resolve()}


def test_convert_config_requires_pdf(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.txt").write_text("hello")

    with pytest.raises(ConfigError):
        ConvertConfig(
            inputs=(tmp_path / "docs",),
            output=tmp_path / "out",
            formats=("md",),
            artifacts_path=tmp_path / ".artifacts",
        )


def test_safe_stem_normalizes_path() -> None:
    assert safe_stem(Path("Demo Report.pdf")) == "Demo_Report"


def test_convert_config_all_formats_keyword(tmp_path: Path) -> None:
    pdf = tmp_path / "alpha.pdf"
    pdf.write_bytes(b"%PDF-1.7 stub")

    config = ConvertConfig(
        inputs=(pdf,),
        output=tmp_path / "out",
        formats=("ALL",),
        artifacts_path=tmp_path / ".artifacts",
    )

    assert config.formats == ALL_FORMATS
