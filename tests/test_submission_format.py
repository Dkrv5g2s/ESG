import csv
import io
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from predict_submission import (  # noqa: E402
    SUBMISSION_COLUMNS,
    build_class_weights,
    build_stratify_labels,
    build_submission_rows,
    default_target_path,
    resolve_device,
    write_submission_csv,
)


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


class FakeTorch:
    __version__ = "2.7.1+cpu"
    version = SimpleNamespace(cuda=None)
    cuda = FakeCuda()

    @staticmethod
    def device(name):
        return SimpleNamespace(type=name)


class SubmissionFormatTest(unittest.TestCase):
    def test_build_class_weights_gives_larger_weight_to_rare_labels(self):
        rows = [
            {"promise_status": "Yes"},
            {"promise_status": "Yes"},
            {"promise_status": "Yes"},
            {"promise_status": "No"},
        ]
        weights = build_class_weights(rows, "promise_status", {"Yes": 0, "No": 1})

        self.assertLess(weights[0], weights[1])

    def test_build_stratify_labels_uses_combined_eval_fields_when_repeated(self):
        rows = [
            {
                "promise_status": "Yes",
                "verification_timeline": "already",
                "evidence_status": "Yes",
                "evidence_quality": "Clear",
            },
            {
                "promise_status": "Yes",
                "verification_timeline": "already",
                "evidence_status": "Yes",
                "evidence_quality": "Clear",
            },
            {
                "promise_status": "No",
                "verification_timeline": "N/A",
                "evidence_status": "N/A",
                "evidence_quality": "N/A",
            },
            {
                "promise_status": "No",
                "verification_timeline": "N/A",
                "evidence_status": "N/A",
                "evidence_quality": "N/A",
            },
        ]

        self.assertEqual(
            build_stratify_labels(rows),
            [
                "Yes|already|Yes|Clear",
                "Yes|already|Yes|Clear",
                "No|N/A|N/A|N/A",
                "No|N/A|N/A|N/A",
            ],
        )

    def test_default_target_path_prefers_validation_json_in_data_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            data_dir = project_root / "data"
            data_dir.mkdir()
            data_target = data_dir / "vpesg4k_val_1000.json"
            root_target = project_root / "vpesg4k_val_1000.json"
            data_target.write_text("[]", encoding="utf-8")
            root_target.write_text("[]", encoding="utf-8")

            self.assertEqual(default_target_path(project_root), data_target)

    def test_resolve_device_rejects_cuda_when_torch_has_no_cuda_support(self):
        with patch.dict(sys.modules, {"torch": FakeTorch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_device("cuda")

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
