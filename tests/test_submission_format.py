import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from predict_submission import (  # noqa: E402
    SUBMISSION_COLUMNS,
    build_submission_rows,
    write_submission_csv,
)


class SubmissionFormatTest(unittest.TestCase):
    def test_build_submission_rows_keeps_official_columns_and_applies_predictions(self):
        source_rows = [
            {
                "id": "11001",
                "data": "測試文字",
                "company": "demo",
            }
        ]
        predictions = [
            {
                "promise_status": "Yes",
                "verification_timeline": "already",
                "evidence_status": "No",
                "evidence_quality": "N/A",
            }
        ]

        rows = build_submission_rows(source_rows, predictions)

        self.assertEqual(list(rows[0].keys()), SUBMISSION_COLUMNS)
        self.assertEqual(rows[0]["id"], "11001")
        self.assertEqual(rows[0]["company"], "demo")
        self.assertEqual(rows[0]["esg_type"], "")
        self.assertEqual(rows[0]["promise_status"], "Yes")
        self.assertEqual(rows[0]["verification_timeline"], "already")
        self.assertEqual(rows[0]["evidence_status"], "No")
        self.assertEqual(rows[0]["evidence_quality"], "N/A")

    def test_write_submission_csv_is_utf8_without_bom_and_uses_lf(self):
        rows = build_submission_rows(
            [{"id": "11001", "data": "測試文字"}],
            [
                {
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "submission.csv"
            write_submission_csv(rows, output_path)
            raw = output_path.read_bytes()

        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn(b"\n", raw)

        decoded = raw.decode("utf-8")
        header = next(csv.reader(io.StringIO(decoded)))
        self.assertEqual(header, SUBMISSION_COLUMNS)


if __name__ == "__main__":
    unittest.main()
