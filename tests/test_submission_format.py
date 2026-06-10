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
    default_validation_path as baseline_default_validation_path,
    default_target_path,
    format_task_name,
    resolve_device,
    TASK_REPORT_ORDER,
    write_submission_csv,
)
from ours import build_class_weights, build_stratify_labels  # noqa: E402
from ours import (  # noqa: E402
    apply_prediction_constraints,
    default_validation_path,
    format_prediction_distribution,
    format_score_report,
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

    def test_build_class_weights_can_smooth_and_cap_extreme_rare_labels(self):
        rows = [{"evidence_quality": "Clear"} for _ in range(552)]
        rows += [{"evidence_quality": "N/A"} for _ in range(323)]
        rows += [{"evidence_quality": "Not Clear"} for _ in range(124)]
        rows += [{"evidence_quality": "Misleading"}]
        label2id = {
            "Clear": 0,
            "Not Clear": 1,
            "Misleading": 2,
            "N/A": 3,
        }

        weights = build_class_weights(
            rows,
            "evidence_quality",
            label2id,
            mode="sqrt",
            max_weight=8.0,
        )

        self.assertGreater(weights[label2id["Misleading"]], weights[label2id["Clear"]])
        self.assertLessEqual(weights[label2id["Misleading"]], 8.0)

    def test_apply_prediction_constraints_keeps_downstream_tasks_consistent(self):
        constrained = apply_prediction_constraints(
            {
                "promise_status": "No",
                "verification_timeline": "already",
                "evidence_status": "Yes",
                "evidence_quality": "Clear",
            }
        )

        self.assertEqual(constrained["verification_timeline"], "N/A")
        self.assertEqual(constrained["evidence_status"], "N/A")
        self.assertEqual(constrained["evidence_quality"], "N/A")

    def test_apply_prediction_constraints_resets_quality_without_evidence(self):
        constrained = apply_prediction_constraints(
            {
                "promise_status": "Yes",
                "verification_timeline": "already",
                "evidence_status": "No",
                "evidence_quality": "Clear",
            }
        )

        self.assertEqual(constrained["verification_timeline"], "already")
        self.assertEqual(constrained["evidence_status"], "No")
        self.assertEqual(constrained["evidence_quality"], "N/A")

    def test_format_score_report_includes_weighted_score_and_task_scores(self):
        report = format_score_report(
            "Best validation result",
            {
                "final_weighted_score": 0.5421278,
                "promise_status": 0.7447,
                "evidence_status": 0.6069,
                "evidence_quality": 0.3976,
                "verification_timeline": 0.4797,
            },
            epoch=9,
        )

        self.assertIn("Best validation result", report)
        self.assertIn("epoch: 9", report)
        self.assertIn("weighted_score: 0.5421", report)
        self.assertIn("promise_status: 0.7447", report)
        self.assertIn("evidence_quality: 0.3976", report)

    def test_format_prediction_distribution_counts_submission_labels(self):
        report = format_prediction_distribution(
            [
                {
                    "promise_status": "Yes",
                    "evidence_status": "Yes",
                    "evidence_quality": "Clear",
                    "verification_timeline": "already",
                },
                {
                    "promise_status": "No",
                    "evidence_status": "N/A",
                    "evidence_quality": "N/A",
                    "verification_timeline": "N/A",
                },
            ]
        )

        self.assertIn("Submission prediction distribution", report)
        self.assertIn("promise_status: Yes=1, No=1", report)
        self.assertIn("evidence_quality: Clear=1, N/A=1", report)

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

    def test_ours_defaults_target_4090_training_without_reusing_old_checkpoint(self):
        with patch.object(sys, "argv", ["ours.py"]):
            args = ours_module.parse_args()

        self.assertEqual(args.model_path, PROJECT_ROOT / "models" / "ours_4090.pt")
        self.assertEqual(args.model_name, "hfl/chinese-roberta-wwm-ext-large")
        self.assertEqual(args.max_len, 512)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.pooling, "mean")
        self.assertEqual(args.class_weight_mode, "sqrt")
        self.assertFalse(args.no_mixed_precision)
        self.assertEqual(args.validation_data, default_validation_path())
        self.assertFalse(hasattr(args, "extra_train_data"))
        self.assertFalse(hasattr(args, "no_final_train_all_data"))

    def test_default_validation_path_prefers_validation_json_in_data_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            data_dir = project_root / "data"
            data_dir.mkdir()
            val_json = data_dir / "vpesg4k_val_1000.json"
            val_csv = data_dir / "vpesg4k_val_1000.csv"
            val_json.write_text("[]", encoding="utf-8")
            val_csv.write_text("", encoding="utf-8")

            self.assertEqual(default_validation_path(project_root), val_json)
            self.assertEqual(baseline_default_validation_path(project_root), val_json)

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

    def test_baseline_main_trains_on_train_rows_validates_on_validation_rows_and_outputs_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.json"
            validation_path = tmp_path / "validation.json"
            target_path = tmp_path / "test.json"
            output_path = tmp_path / "submission.csv"
            model_path = tmp_path / "missing-model.pt"
            train_path.write_text(
                """[
                    {
                        "id": "train",
                        "data": "訓練",
                        "promise_status": "No",
                        "verification_timeline": "N/A",
                        "evidence_status": "N/A",
                        "evidence_quality": "N/A"
                    }
                ]""",
                encoding="utf-8",
            )
            validation_path.write_text(
                """[
                    {
                        "id": "validation",
                        "data": "驗證",
                        "promise_status": "No",
                        "verification_timeline": "N/A",
                        "evidence_status": "N/A",
                        "evidence_quality": "N/A"
                    }
                ]""",
                encoding="utf-8",
            )
            target_path.write_text('[{"id": "test", "data": "測試"}]', encoding="utf-8")

            argv = [
                "baseline_reference.py",
                "--target",
                str(target_path),
                "--train-data",
                str(train_path),
                "--validation-data",
                str(validation_path),
                "--output",
                str(output_path),
                "--model-path",
                str(model_path),
                "--no-download-train",
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
                "train_and_predict",
                return_value=predictions,
            ) as train_mock:
                baseline_reference.main()

            train_args = train_mock.call_args.args
            self.assertEqual(train_args[0][0]["id"], "train")
            self.assertEqual(train_args[1][0]["id"], "validation")
            self.assertEqual(train_args[2][0]["id"], "test")
            self.assertTrue(output_path.exists())

    def test_ours_main_trains_on_train_rows_validates_on_validation_rows_and_outputs_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.json"
            validation_path = tmp_path / "validation.json"
            target_path = tmp_path / "test.json"
            output_path = tmp_path / "submission.csv"
            model_path = tmp_path / "missing-model.pt"
            train_path.write_text(
                """[
                    {
                        "id": "train",
                        "data": "訓練",
                        "promise_status": "No",
                        "verification_timeline": "N/A",
                        "evidence_status": "N/A",
                        "evidence_quality": "N/A"
                    }
                ]""",
                encoding="utf-8",
            )
            validation_path.write_text(
                """[
                    {
                        "id": "validation",
                        "data": "驗證",
                        "promise_status": "No",
                        "verification_timeline": "N/A",
                        "evidence_status": "N/A",
                        "evidence_quality": "N/A"
                    }
                ]""",
                encoding="utf-8",
            )
            target_path.write_text('[{"id": "test", "data": "測試"}]', encoding="utf-8")

            argv = [
                "ours.py",
                "--target",
                str(target_path),
                "--train-data",
                str(train_path),
                "--validation-data",
                str(validation_path),
                "--output",
                str(output_path),
                "--model-path",
                str(model_path),
                "--no-download-train",
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
                ours_module,
                "train_and_predict",
                return_value=predictions,
            ) as train_mock:
                ours_module.main()

            train_args = train_mock.call_args.args
            self.assertEqual(train_args[0][0]["id"], "train")
            self.assertEqual(train_args[1][0]["id"], "validation")
            self.assertEqual(train_args[2][0]["id"], "test")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
