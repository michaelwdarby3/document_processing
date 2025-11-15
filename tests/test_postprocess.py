from pathlib import Path

import csv
import json

from docling_offline.postprocess import export_tables_to_csv


def test_export_tables_to_csv(tmp_path: Path) -> None:
    doc = {
        "tables": [
            {
                "label": "table",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {"l": 0, "t": 40, "r": 60, "b": 0, "coord_origin": "BOTTOMLEFT"},
                    }
                ],
                "data": {
                    "grid": [
                        [
                            {"text": "Header A", "column_header": True},
                            {"text": "Header B", "column_header": True},
                        ],
                        [
                            {"text": "Row 1", "row_header": True},
                            {"text": "Value 1"},
                        ],
                        [
                            {"text": "Row 2", "row_header": True},
                            {"text": "Value 2"},
                        ],
                    ]
                }
            }
        ],
        "texts": [
            {
                "label": "section_header",
                "text": "Demo Heading",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {"l": 0, "t": 55, "r": 60, "b": 45, "coord_origin": "BOTTOMLEFT"},
                    }
                ],
            }
        ],
    }

    json_path = tmp_path / "doc.json"
    json_path.write_text(json.dumps(doc), encoding="utf-8")
    out_dir = tmp_path / "tables"

    outputs = export_tables_to_csv(json_path, out_dir)

    assert len(outputs) == 1
    csv_path = outputs[0]
    assert csv_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["Demo Heading"]
    assert rows[1] == []
    assert rows[2] == ["[COLUMN HEADER] Header A", "[COLUMN HEADER] Header B"]
    assert rows[3] == ["[ROW HEADER] Row 1", "Value 1"]
