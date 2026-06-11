import csv
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_runner as runner  # noqa: E402


class ExperimentRunnerTest(unittest.TestCase):
    def test_runner_is_standalone_and_does_not_import_ours(self):
        source = (PROJECT_ROOT / "experiment_runner.py").read_text(encoding="utf-8")

        self.assertNotIn("from ours import", source)

    def test_default_configs_include_merge_reference_settings(self):
        configs = {config.name: config for config in runner.default_experiment_configs()}

        self.assertIn("balanced_cap3_lr3e-5_do0.10_ls0.00_c1", configs)
        config = configs["balanced_cap3_lr3e-5_do0.10_ls0.00_c1"]
        self.assertEqual(config.class_weight_mode, "balanced")
        self.assertEqual(config.max_class_weight, 3.0)
        self.assertEqual(config.learning_rate, 3e-5)
        self.assertEqual(config.dropout_rate, 0.10)
        self.assertEqual(config.label_smoothing, 0.0)
        self.assertTrue(config.apply_prediction_constraints)
        self.assertEqual(config.final_epochs, 8)

    def test_build_search_command_uses_validation_only_flow(self):
        config = runner.ExperimentConfig(
            name="demo",
            class_weight_mode="balanced",
            max_class_weight=3.0,
            learning_rate=3e-5,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            command = runner.build_ours_command(
                python_exe="python",
                project_root=PROJECT_ROOT,
                config=config,
                seed=42,
                run_dir=run_dir,
                device="cuda",
                merge_for_submission=False,
            )

        self.assertIn("--force-train", command)
        self.assertIn(str(PROJECT_ROOT / "experiment_runner.py"), command)
        self.assertIn("--mode", command)
        self.assertIn("single", command)
        self.assertIn("--no-merge-train-val-for-submission", command)
        self.assertNotIn("--final-epochs", command)
        self.assertIn("--apply-prediction-constraints", command)
        self.assertIn("--device", command)
        self.assertIn("cuda", command)

    def test_build_final_command_uses_merge_flow_and_fixed_final_epochs(self):
        config = runner.ExperimentConfig(
            name="demo",
            class_weight_mode="balanced",
            max_class_weight=3.0,
            learning_rate=3e-5,
            final_epochs=8,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            command = runner.build_ours_command(
                python_exe="python",
                project_root=PROJECT_ROOT,
                config=config,
                seed=42,
                run_dir=run_dir,
                device="cuda",
                merge_for_submission=True,
            )

        self.assertIn("--merge-train-val-for-submission", command)
        self.assertIn("--final-epochs", command)
        self.assertIn("8", command)
        self.assertNotIn("--no-merge-train-val-for-submission", command)

    def test_summarize_results_sorts_by_robust_score(self):
        configs = {
            "stable": runner.ExperimentConfig(name="stable"),
            "spiky": runner.ExperimentConfig(name="spiky"),
        }
        results = [
            runner.ExperimentResult(config_name="stable", seed=1, status="ok", weighted_score=0.60),
            runner.ExperimentResult(config_name="stable", seed=2, status="ok", weighted_score=0.59),
            runner.ExperimentResult(config_name="spiky", seed=1, status="ok", weighted_score=0.62),
            runner.ExperimentResult(config_name="spiky", seed=2, status="ok", weighted_score=0.54),
        ]

        summary = runner.summarize_results(results, configs)

        self.assertEqual(summary[0]["config_name"], "stable")
        self.assertGreater(summary[0]["robust_score"], summary[1]["robust_score"])

    def test_weighted_vote_submission_applies_consistency_constraints(self):
        rows_a = [
            {
                "id": "1",
                "data": "demo",
                "promise_status": "No",
                "verification_timeline": "already",
                "evidence_status": "Yes",
                "evidence_quality": "Clear",
            }
        ]
        rows_b = [
            {
                "id": "1",
                "data": "demo",
                "promise_status": "Yes",
                "verification_timeline": "already",
                "evidence_status": "Yes",
                "evidence_quality": "Clear",
            }
        ]

        voted = runner.weighted_vote_submission([rows_a, rows_b], [0.7, 0.2])

        self.assertEqual(voted[0]["promise_status"], "No")
        self.assertEqual(voted[0]["verification_timeline"], "N/A")
        self.assertEqual(voted[0]["evidence_status"], "N/A")
        self.assertEqual(voted[0]["evidence_quality"], "N/A")

    def test_write_summary_csv_contains_ranked_rows(self):
        rows = [
            {
                "config_name": "demo",
                "runs": 1,
                "mean_weighted": 0.6,
                "std_weighted": 0.0,
                "robust_score": 0.6,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "summary.csv"
            runner.write_summary_csv(rows, output_path)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))

        self.assertEqual(loaded[0]["config_name"], "demo")
        self.assertEqual(loaded[0]["rank"], "1")


if __name__ == "__main__":
    unittest.main()
