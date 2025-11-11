from enum import Enum
from pathlib import Path
from typing import Dict

import pytest

from docling_offline import processor


class StubDocument:
    def __init__(self, source: Path):
        self.source = source

    def export_to_markdown(self) -> str:
        return f"# {self.source.name}"

    def export_to_text(self) -> str:
        return f"{self.source.name}"

    def export_to_dict(self) -> Dict[str, str]:
        return {"doc": self.source.name}

    def export_to_document_tokens(self) -> str:
        return f"doctags:{self.source.name}"


class StubConversionStatus(Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class StubConversionResult:
    def __init__(self, source: Path):
        self.status = StubConversionStatus.SUCCEEDED
        self.document = StubDocument(source)


class StubFormatOption:
    def __init__(self, backend, pipeline_cls, pipeline_options):
        self.backend = backend
        self.pipeline_cls = pipeline_cls
        self.pipeline_options = pipeline_options


class StubPdfPipelineOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def model_copy(self, update=None):
        data = dict(self.kwargs)
        if update:
            data.update(update)
        return StubPdfPipelineOptions(**data)


class StubEasyOcrOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubTesseractCliOcrOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubTesseractOcrOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubPdfDocumentBackend:
    pass


class StubTableStructureOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def model_copy(self, update=None):
        data = dict(self.kwargs)
        if update:
            data.update(update)
        return StubTableStructureOptions(**data)


class StubTableFormerMode:
    FAST = "fast"
    ACCURATE = "accurate"


class StubStandardPdfPipeline:
    calls = 0

    @staticmethod
    def download_models_hf(local_dir: Path, force: bool = False) -> Path:
        StubStandardPdfPipeline.calls += 1
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir

    @staticmethod
    def get_default_options() -> StubPdfPipelineOptions:
        return StubPdfPipelineOptions()


class StubInputFormat:
    PDF = "pdf"


class StubDocumentConverter:
    def __init__(self, format_options: Dict):
        self.format_options = format_options

    def convert(self, source):
        return StubConversionResult(Path(source))


@pytest.fixture(autouse=True)
def stub_docling(monkeypatch: pytest.MonkeyPatch) -> Dict[str, object]:
    """Patch processor._load_docling to avoid pulling real Docling in unit tests."""

    StubStandardPdfPipeline.calls = 0

    resources = {
        "DocumentConverter": StubDocumentConverter,
        "FormatOption": StubFormatOption,
        "InputFormat": StubInputFormat,
        "PdfPipelineOptions": StubPdfPipelineOptions,
        "PdfBackend": StubPdfDocumentBackend,
        "StandardPdfPipeline": StubStandardPdfPipeline,
        "EasyOcrOptions": StubEasyOcrOptions,
        "TesseractCliOcrOptions": StubTesseractCliOcrOptions,
        "TesseractOcrOptions": StubTesseractOcrOptions,
        "ConversionStatus": StubConversionStatus,
        "TableStructureOptions": StubTableStructureOptions,
        "TableFormerMode": StubTableFormerMode,
        "snapshot_download": None,
        "Path": Path,
    }

    monkeypatch.setattr(processor, "_load_docling", lambda: resources)
    return resources
