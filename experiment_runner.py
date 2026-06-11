from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from ours import (
    EVAL_FIELDS,
    SUBMISSION_COLUMNS,
    apply_prediction_constraints,
    build_stratify_labels,
    load_rows,
    normalize_value,
    write_submission_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [42, 43, 44]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_len: int = 256
    batch_size: int = 8
    epochs: int = 15
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout_rate: float = 0.1
    head_hidden_size: int = 0
    early_stopping_patience: int = 5
    min_delta: float = 0.0
    pooling: str = "cls"
    class_weight_mode: str = "balanced"
    max_class_weight: float = 3.0
    label_smoothing: float = 0.0
    apply_prediction_constraints: bool = True
    final_epochs: int = 8


@dataclass(frozen=True)
class ExperimentResult:
    config_name: str
    seed: int
    status: str
    weighted_score: float | None = None
    best_epoch: int | None = None
    promise_status: float | None = None
    evidence_status: float | None = None
    evidence_quality: float | None = None
    verification_timeline: float | None = None
    run_dir: str = ""
    log_path: str = ""
    message: str = ""


def compact_float(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace(".", "p").replace("-", "m")


def format_lr(value: float) -> str:
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"{mantissa}e{int(exponent)}"


def format_cap(value: float) -> str:
    if value <= 0:
        return "none"
    if float(value).is_integer():
        return str(int(value))
    return compact_float(value)


def make_config_name(
    *,
    class_weight_mode: str,
    max_class_weight: float,
    learning_rate: float,
    dropout_rate: float,
    label_smoothing: float,
    apply_prediction_constraints: bool,
    max_len: int = 256,
    head_hidden_size: int = 0,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> str:
    parts = [
        class_weight_mode,
        f"cap{format_cap(max_class_weight)}",
        f"lr{format_lr(learning_rate)}",
        f"do{dropout_rate:.2f}",
        f"ls{label_smoothing:.2f}",
        f"c{1 if apply_prediction_constraints else 0}",
    ]
    if max_len != 256:
        parts.append(f"len{max_len}")
    if head_hidden_size:
        parts.append(f"head{head_hidden_size}")
    if warmup_ratio != 0.1:
        parts.append(f"wu{compact_float(warmup_ratio)}")
    if weight_decay != 0.01:
        parts.append(f"wd{compact_float(weight_decay)}")
    return "_".join(parts)


def default_experiment_configs(profile: str = "full") -> list[ExperimentConfig]:
    if profile == "quick":
        class_variants = [("balanced", 3.0), ("balanced", 4.0), ("sqrt", 4.0)]
        learning_rates = [2.5e-5, 3e-5, 3.5e-5]
        dropouts = [0.1, 0.15]
        label_smoothing_values = [0.0]
        constraints = [True]
        max_lens = [256]
        head_sizes = [0]
        warmups = [0.1]
        weight_decays = [0.01]
    elif profile == "huge":
        class_variants = [
            ("balanced", 2.0),
            ("balanced", 2.5),
            ("balanced", 3.0),
            ("balanced", 3.5),
            ("balanced", 4.0),
            ("balanced", 5.0),
            ("sqrt", 3.0),
            ("sqrt", 4.0),
            ("sqrt", 6.0),
            ("sqrt", 8.0),
            ("none", 0.0),
        ]
        learning_rates = [1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5]
        dropouts = [0.05, 0.1, 0.15, 0.2]
        label_smoothing_values = [0.0, 0.02, 0.04]
        constraints = [True, False]
        max_lens = [256, 384]
        head_sizes = [0, 128]
        warmups = [0.06, 0.1, 0.15]
        weight_decays = [0.01, 0.02]
    else:
        class_variants = [
            ("balanced", 2.5),
            ("balanced", 3.0),
            ("balanced", 3.5),
            ("balanced", 4.0),
            ("sqrt", 4.0),
            ("sqrt", 6.0),
        ]
        learning_rates = [2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5]
        dropouts = [0.05, 0.1, 0.15]
        label_smoothing_values = [0.0, 0.03]
        constraints = [True, False]
        max_lens = [256]
        head_sizes = [0]
        warmups = [0.1]
        weight_decays = [0.01]

    configs: dict[str, ExperimentConfig] = {}
    for class_weight_mode, max_class_weight in class_variants:
        for learning_rate in learning_rates:
            for dropout_rate in dropouts:
                for label_smoothing in label_smoothing_values:
                    for apply_constraints in constraints:
                        for max_len in max_lens:
                            for head_hidden_size in head_sizes:
                                for warmup_ratio in warmups:
                                    for weight_decay in weight_decays:
                                        name = make_config_name(
                                            class_weight_mode=class_weight_mode,
                                            max_class_weight=max_class_weight,
                                            learning_rate=learning_rate,
                                            dropout_rate=dropout_rate,
                                            label_smoothing=label_smoothing,
                                            apply_prediction_constraints=apply_constraints,
                                            max_len=max_len,
                                            head_hidden_size=head_hidden_size,
                                            warmup_ratio=warmup_ratio,
                                            weight_decay=weight_decay,
                                        )
                                        configs[name] = ExperimentConfig(
                                            name=name,
                                            max_len=max_len,
                                            learning_rate=learning_rate,
                                            weight_decay=weight_decay,
                                            warmup_ratio=warmup_ratio,
                                            dropout_rate=dropout_rate,
                                            head_hidden_size=head_hidden_size,
                                            class_weight_mode=class_weight_mode,
                                            max_class_weight=max_class_weight,
                                            label_smoothing=label_smoothing,
                                            apply_prediction_constraints=apply_constraints,
                                        )

    return sorted(configs.values(), key=lambda config: config.name)


def experiment_config_to_cli_args(config: ExperimentConfig) -> list[str]:
    args = [
        "--model-name",
        config.model_name,
        "--max-len",
        str(config.max_len),
        "--batch-size",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--warmup-ratio",
        str(config.warmup_ratio),
        "--dropout-rate",
        str(config.dropout_rate),
        "--head-hidden-size",
        str(config.head_hidden_size),
        "--early-stopping-patience",
        str(config.early_stopping_patience),
        "--min-delta",
        str(config.min_delta),
        "--pooling",
        config.pooling,
        "--class-weight-mode",
        config.class_weight_mode,
        "--max-class-weight",
        str(config.max_class_weight),
        "--label-smoothing",
        str(config.label_smoothing),
    ]
    if config.class_weight_mode == "none":
        args.append("--no-class-weights")
    if config.apply_prediction_constraints:
        args.append("--apply-prediction-constraints")
    else:
        args.append("--no-apply-prediction-constraints")
    return args


def build_ours_command(
    *,
    python_exe: str,
    project_root: Path,
    config: ExperimentConfig,
    seed: int,
    run_dir: Path,
    device: str,
    merge_for_submission: bool,
    train_data: Path | None = None,
    validation_data: Path | None = None,
    target_data: Path | None = None,
) -> list[str]:
    train_data = train_data or project_root / "data" / "vpesg4k_train_1000.json"
    validation_data = validation_data or project_root / "data" / "vpesg4k_val_1000.json"
    target_data = target_data or project_root / "data" / "vpesg4k_test_2000.json"
    model_path = run_dir / "model.pt"
    output_path = run_dir / "submission.csv"
    metrics_path = run_dir / "validation_metrics.json"

    command = [
        python_exe,
        str(project_root / "ours.py"),
        "--target",
        str(target_data),
        "--train-data",
        str(train_data),
        "--validation-data",
        str(validation_data),
        "--output",
        str(output_path),
        "--metrics-output",
        str(metrics_path),
        "--model-path",
        str(model_path),
        "--seed",
        str(seed),
        "--device",
        device,
        "--force-train",
        "--no-download-train",
    ]
    command.extend(experiment_config_to_cli_args(config))
    if merge_for_submission:
        command.extend(
            [
                "--merge-train-val-for-submission",
                "--final-epochs",
                str(config.final_epochs),
            ]
        )
    else:
        command.append("--no-merge-train-val-for-submission")
    return command


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_result(
    *,
    config_name: str,
    seed: int,
    run_dir: Path,
    return_code: int,
    log_path: Path,
) -> ExperimentResult:
    metrics_path = run_dir / "validation_metrics.json"
    if return_code != 0:
        return ExperimentResult(
            config_name=config_name,
            seed=seed,
            status="failed",
            run_dir=str(run_dir),
            log_path=str(log_path),
            message=f"return_code={return_code}",
        )
    if not metrics_path.exists():
        return ExperimentResult(
            config_name=config_name,
            seed=seed,
            status="missing_metrics",
            run_dir=str(run_dir),
            log_path=str(log_path),
            message="validation_metrics.json was not written",
        )

    metrics = read_metrics(metrics_path)
    task_f1 = metrics.get("task_f1") or {}
    return ExperimentResult(
        config_name=config_name,
        seed=seed,
        status="ok",
        weighted_score=metrics.get("weighted_score"),
        best_epoch=metrics.get("best_epoch"),
        promise_status=task_f1.get("promise_status"),
        evidence_status=task_f1.get("evidence_status"),
        evidence_quality=task_f1.get("evidence_quality"),
        verification_timeline=task_f1.get("verification_timeline"),
        run_dir=str(run_dir),
        log_path=str(log_path),
    )


def result_to_row(result: ExperimentResult) -> dict[str, Any]:
    return {field.name: getattr(result, field.name) for field in fields(ExperimentResult)}


def write_results_csv(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(ExperimentResult)])
        writer.writeheader()
        writer.writerows(result_to_row(result) for result in results)


def summarize_results(
    results: list[ExperimentResult],
    configs: dict[str, ExperimentConfig],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExperimentResult]] = defaultdict(list)
    for result in results:
        if result.status == "ok" and result.weighted_score is not None:
            grouped[result.config_name].append(result)

    summary = []
    for config_name, group in grouped.items():
        scores = [float(result.weighted_score) for result in group if result.weighted_score is not None]
        if not scores:
            continue
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        row = {
            "config_name": config_name,
            "runs": len(scores),
            "mean_weighted": statistics.fmean(scores),
            "std_weighted": std,
            "min_weighted": min(scores),
            "max_weighted": max(scores),
            "robust_score": statistics.fmean(scores) - std,
        }
        for metric in (
            "promise_status",
            "evidence_status",
            "evidence_quality",
            "verification_timeline",
        ):
            metric_scores = [
                float(getattr(result, metric))
                for result in group
                if getattr(result, metric) is not None
            ]
            row[f"mean_{metric}"] = statistics.fmean(metric_scores) if metric_scores else ""
        config = configs.get(config_name)
        if config is not None:
            row.update(
                {
                    "class_weight_mode": config.class_weight_mode,
                    "max_class_weight": config.max_class_weight,
                    "learning_rate": config.learning_rate,
                    "dropout_rate": config.dropout_rate,
                    "label_smoothing": config.label_smoothing,
                    "apply_prediction_constraints": config.apply_prediction_constraints,
                    "max_len": config.max_len,
                    "head_hidden_size": config.head_hidden_size,
                    "final_epochs": config.final_epochs,
                }
            )
        summary.append(row)

    summary.sort(
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_weighted"]),
            float(row["min_weighted"]),
        ),
        reverse=True,
    )
    return summary


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            output = dict(row)
            output["rank"] = index
            writer.writerow(output)


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_submission_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_vote_submission(
    submissions: list[list[dict[str, str]]],
    weights: list[float],
) -> list[dict[str, str]]:
    if not submissions:
        raise ValueError("At least one submission is required for weighted voting.")
    if len(submissions) != len(weights):
        raise ValueError("Submission and weight counts differ.")

    row_count = len(submissions[0])
    for submission in submissions:
        if len(submission) != row_count:
            raise ValueError("Submission row counts differ.")

    voted_rows = []
    for row_index in range(row_count):
        output = {
            column: normalize_value(submissions[0][row_index].get(column, ""))
            for column in SUBMISSION_COLUMNS
        }
        prediction = {}
        for field, labels in EVAL_FIELDS.items():
            scores = Counter()
            for submission, weight in zip(submissions, weights):
                value = normalize_value(submission[row_index].get(field, ""))
                if value in labels:
                    scores[value] += float(weight)
            prediction[field] = max(labels, key=lambda label: (scores[label], -labels.index(label)))
        output.update(apply_prediction_constraints(prediction))
        voted_rows.append(output)
    return voted_rows


def write_config_manifest(configs: list[ExperimentConfig], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in configs:
        rows.append({field.name: getattr(config, field.name) for field in fields(ExperimentConfig)})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(ExperimentConfig)])
        writer.writeheader()
        writer.writerows(rows)


def make_holdout_split(
    train_data: Path,
    output_dir: Path,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[Path, Path]:
    rows = load_rows(train_data)
    labels = build_stratify_labels(rows)
    if labels is None:
        labels = ["__all__"] * len(rows)

    import random

    random.seed(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[label].append(row)

    train_rows = []
    validation_rows = []
    for group_rows in grouped.values():
        shuffled = list(group_rows)
        random.shuffle(shuffled)
        if len(shuffled) < 3:
            train_rows.extend(shuffled)
            continue
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
        validation_rows.extend(shuffled[:validation_count])
        train_rows.extend(shuffled[validation_count:])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"holdout_train_seed{seed}.json"
    validation_path = output_dir / f"holdout_val_seed{seed}.json"
    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_path.write_text(
        json.dumps(validation_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_path, validation_path


def run_command(command: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode


def run_search(args: argparse.Namespace, configs: list[ExperimentConfig], run_root: Path) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    split_root = run_root / "splits"
    for config_index, config in enumerate(configs, start=1):
        for seed in args.seeds:
            run_dir = run_root / "search" / config.name / f"seed{seed}"
            log_path = run_dir / "train.log"
            if args.validation_strategy == "holdout":
                train_data, validation_data = make_holdout_split(
                    args.train_data,
                    split_root,
                    seed=seed,
                    validation_fraction=args.holdout_fraction,
                )
            else:
                train_data = args.train_data
                validation_data = args.validation_data

            command = build_ours_command(
                python_exe=args.python_exe,
                project_root=args.project_root,
                config=config,
                seed=seed,
                run_dir=run_dir,
                device=args.device,
                merge_for_submission=False,
                train_data=train_data,
                validation_data=validation_data,
                target_data=args.target_data,
            )
            print(
                f"[search {config_index}/{len(configs)} seed={seed}] "
                f"{config.name}"
            )
            return_code = run_command(
                command,
                cwd=args.project_root,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            result = parse_result(
                config_name=config.name,
                seed=seed,
                run_dir=run_dir,
                return_code=return_code,
                log_path=log_path,
            )
            results.append(result)
            write_results_csv(results, run_root / "search_results.csv")
            summary = summarize_results(results, {config.name: config for config in configs})
            write_summary_csv(summary, run_root / "search_summary.csv")
            if result.status == "ok":
                print(
                    f"  score={result.weighted_score:.5f} "
                    f"best_epoch={result.best_epoch}"
                )
            else:
                print(f"  status={result.status}; see {log_path}")
    return results


def select_configs_from_summary(
    summary_rows: list[dict[str, Any]],
    configs: dict[str, ExperimentConfig],
    *,
    top_k: int,
) -> list[tuple[ExperimentConfig, float]]:
    selected = []
    for row in summary_rows[:top_k]:
        config = configs.get(row["config_name"])
        if config is None:
            continue
        weight = float(row.get("robust_score") or row.get("mean_weighted") or 1.0)
        selected.append((config, max(weight, 1e-6)))
    return selected


def run_final(
    args: argparse.Namespace,
    configs: dict[str, ExperimentConfig],
    run_root: Path,
    summary_path: Path,
) -> None:
    summary_rows = read_summary_csv(summary_path)
    selected = select_configs_from_summary(
        summary_rows,
        configs,
        top_k=args.final_top_k,
    )
    if not selected:
        raise ValueError(f"No final configs selected from {summary_path}")

    submissions = []
    weights = []
    for config, robust_weight in selected:
        for seed in args.final_seeds:
            run_dir = run_root / "final" / config.name / f"seed{seed}"
            log_path = run_dir / "train.log"
            command = build_ours_command(
                python_exe=args.python_exe,
                project_root=args.project_root,
                config=config,
                seed=seed,
                run_dir=run_dir,
                device=args.device,
                merge_for_submission=True,
                train_data=args.train_data,
                validation_data=args.validation_data,
                target_data=args.target_data,
            )
            print(f"[final seed={seed}] {config.name} weight={robust_weight:.5f}")
            return_code = run_command(
                command,
                cwd=args.project_root,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            if return_code != 0:
                print(f"  failed; see {log_path}")
                continue
            submission_path = run_dir / "submission.csv"
            if submission_path.exists():
                submissions.append(load_submission_csv(submission_path))
                weights.append(robust_weight)

    if submissions:
        ensemble_rows = weighted_vote_submission(submissions, weights)
        ensemble_path = run_root / "final" / "submission_ensemble.csv"
        write_submission_csv(ensemble_rows, ensemble_path)
        print(f"Weighted ensemble submission written to: {ensemble_path}")
    else:
        print("No final submissions were available for ensembling.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run many reproducible VeriPromiseESG experiments, rank by validation "
            "stability, then create final train+val submissions."
        )
    )
    parser.add_argument("--mode", choices=["search", "final", "all"], default="search")
    parser.add_argument("--profile", choices=["quick", "full", "huge"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="Limit configs after deterministic sorting.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train-data", type=Path, default=PROJECT_ROOT / "data" / "vpesg4k_train_1000.json")
    parser.add_argument("--validation-data", type=Path, default=PROJECT_ROOT / "data" / "vpesg4k_val_1000.json")
    parser.add_argument("--target-data", type=Path, default=PROJECT_ROOT / "data" / "vpesg4k_test_2000.json")
    parser.add_argument(
        "--validation-strategy",
        choices=["official", "holdout"],
        default="official",
        help="Use official validation, or split train data for a coarse anti-overfit search.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None, help="Summary CSV for --mode final.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root or args.project_root / "experiments" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    configs = default_experiment_configs(args.profile)
    if args.limit is not None:
        configs = configs[: args.limit]
    config_map = {config.name: config for config in configs}
    write_config_manifest(configs, run_root / "configs.csv")
    print(f"Run root: {run_root}")
    print(f"Configs: {len(configs)}; search seeds: {args.seeds}; final seeds: {args.final_seeds}")
    print(f"Validation strategy: {args.validation_strategy}")

    summary_path = args.summary
    if args.mode in {"search", "all"}:
        results = run_search(args, configs, run_root)
        summary_rows = summarize_results(results, config_map)
        summary_path = run_root / "search_summary.csv"
        write_summary_csv(summary_rows, summary_path)
        print(f"Search summary written to: {summary_path}")
        for row in summary_rows[: min(10, len(summary_rows))]:
            print(
                f"  #{row.get('rank', '?')} {row['config_name']} "
                f"robust={float(row['robust_score']):.5f} "
                f"mean={float(row['mean_weighted']):.5f} "
                f"std={float(row['std_weighted']):.5f}"
            )

    if args.mode in {"final", "all"}:
        if summary_path is None:
            raise ValueError("--summary is required when running --mode final")
        run_final(args, config_map, run_root, summary_path)


if __name__ == "__main__":
    main()
