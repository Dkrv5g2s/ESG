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

import baseline_reference  # noqa: E402
import ours as ours_module  # noqa: E402
from baseline_reference import (  # noqa: E402
    SUBMISSION_COLUMNS,
    build_submission_rows,
    default_target_path,
    format_task_name,
    resolve_device,
    TASK_REPORT_ORDER,
    write_submission_csv,
)
from ours import build_class_weights, build_stratify_labels  # noqa: E402


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
    def test_training_report_uses_official_task_names_in_official_order(self):
        self.assertEqual(
            [(field, format_task_name(field)) for field in TASK_REPORT_ORDER],
            [
                ("promise_status", "Commitment Classification"),
                ("evidence_status", "Evidence Identification"),
                ("evidence_quality", "Clarity Classification"),
                ("verification_timeline", "Timeline Classification"),
            ],
        )

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

    def test_default_target_path_prefers_test_json_in_data_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            data_dir = project_root / "data"
            data_dir.mkdir()
            test_target = data_dir / "vpesg4k_test_2000.json"
            val_target = data_dir / "vpesg4k_val_1000.json"
            test_target.write_text("[]", encoding="utf-8")
            val_target.write_text("[]", encoding="utf-8")

            self.assertEqual(default_target_path(project_root), test_target)
            self.assertEqual(ours_module.default_target_path(project_root), test_target)

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

    def test_main_uses_existing_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_path = tmp_path / "target.json"
            output_path = tmp_path / "submission.csv"
            model_path = tmp_path / "model.pt"
            target_path.write_text('[{"id": "11001", "data": "測試文字"}]', encoding="utf-8")
            model_path.write_bytes(b"placeholder")

            argv = [
                "baseline_reference.py",
                "--target",
                str(target_path),
                "--output",
                str(output_path),
                "--model-path",
                str(model_path),
            ]
            predictions = [
                {
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            ]

            with patch.object(sys, "argv", argv), patch.object(
                baseline_reference,
                "predict_with_checkpoint",
                return_value=predictions,
            ) as predict_mock, patch.object(
                baseline_reference,
                "train_and_predict",
            ) as train_mock:
                baseline_reference.main()

            predict_mock.assert_called_once()
            train_mock.assert_not_called()
            self.assertTrue(output_path.exists())

    def test_main_trains_when_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_path = tmp_path / "target.json"
            train_path = tmp_path / "train.json"
            output_path = tmp_path / "submission.csv"
            model_path = tmp_path / "missing-model.pt"
            target_path.write_text('[{"id": "11001", "data": "測試文字"}]', encoding="utf-8")
            train_path.write_text(
                """[
                    {
                        "id": "1",
                        "data": "訓練文字",
                        "promise_status": "No",
                        "verification_timeline": "N/A",
                        "evidence_status": "N/A",
                        "evidence_quality": "N/A"
                    }
                ]""",
                encoding="utf-8",
            )

            argv = [
                "baseline_reference.py",
                "--target",
                str(target_path),
                "--train-data",
                str(train_path),
                "--output",
                str(output_path),
                "--model-path",
                str(model_path),
            ]
            predictions = [
                {
                    "promise_status": "No",
                    "verification_timeline": "N/A",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                }
            ]

            with patch.object(sys, "argv", argv), patch.object(
                baseline_reference,
                "download_train_data",
            ) as download_mock, patch.object(
                baseline_reference,
                "train_and_predict",
                return_value=predictions,
            ) as train_mock, patch.object(
                baseline_reference,
                "predict_with_checkpoint",
            ) as predict_mock:
                baseline_reference.main()

            download_mock.assert_called_once_with(train_path)
            train_mock.assert_called_once()
            predict_mock.assert_not_called()
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
