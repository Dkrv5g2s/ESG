from __future__ import annotations

import argparse
import csv
import json
import math
import random
import urllib.request
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DATA_URL = (
    "https://raw.githubusercontent.com/veripromiseesg/veripromiseesgdataset/"
    "refs/heads/main/vpesg4k_train_1000.json"
)

SUBMISSION_COLUMNS = [
    "id",
    "data",
    "esg_type",
    "promise_status",
    "promise_string",
    "verification_timeline",
    "evidence_status",
    "evidence_string",
    "evidence_quality",
    "company",
    "ticker",
    "page_number",
    "pdf_url",
    "company_source",
]

EVAL_FIELDS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

FIELD_WEIGHTS = {
    "promise_status": 0.2,
    "verification_timeline": 0.15,
    "evidence_status": 0.3,
    "evidence_quality": 0.35,
}

TASK_REPORT_ORDER = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]

TASK_DISPLAY_NAMES = {
    "promise_status": "Commitment Classification",
    "evidence_status": "Evidence Identification",
    "evidence_quality": "Clarity Classification",
    "verification_timeline": "Timeline Classification",
}

DEFAULT_PREDICTIONS = {
    "promise_status": "No",
    "verification_timeline": "N/A",
    "evidence_status": "N/A",
    "evidence_quality": "N/A",
}


def build_label_maps() -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    label2id = {
        field: {label: idx for idx, label in enumerate(labels)}
        for field, labels in EVAL_FIELDS.items()
    }
    id2label = {
        field: {idx: label for idx, label in enumerate(labels)}
        for field, labels in EVAL_FIELDS.items()
    }
    return label2id, id2label


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return rows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_rows(path)
    if suffix == ".csv":
        return load_csv_rows(path)
    raise ValueError(f"Only .json and .csv inputs are supported: {path}")


def download_train_data(path: Path, url: str = TRAIN_DATA_URL) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def copy_existing_or_default_predictions(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    predictions = []
    for row in rows:
        prediction = {}
        for field, labels in EVAL_FIELDS.items():
            value = normalize_value(row.get(field, ""))
            prediction[field] = value if value in labels else DEFAULT_PREDICTIONS[field]
        predictions.append(apply_prediction_constraints(prediction))
    return predictions


def apply_prediction_constraints(prediction: dict[str, Any]) -> dict[str, str]:
    constrained = {
        field: normalize_value(prediction.get(field, DEFAULT_PREDICTIONS[field]))
        for field in EVAL_FIELDS
    }
    for field, labels in EVAL_FIELDS.items():
        if constrained[field] not in labels:
            constrained[field] = DEFAULT_PREDICTIONS[field]

    if constrained["promise_status"] == "No":
        constrained["verification_timeline"] = "N/A"
        constrained["evidence_status"] = "N/A"
        constrained["evidence_quality"] = "N/A"
        return constrained

    if constrained["evidence_status"] in {"No", "N/A"}:
        constrained["evidence_quality"] = "N/A"
    return constrained


def build_submission_rows(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if len(source_rows) != len(predictions):
        raise ValueError(
            f"Row count mismatch: source={len(source_rows)}, predictions={len(predictions)}"
        )

    rows = []
    for source, prediction in zip(source_rows, predictions):
        output = {column: normalize_value(source.get(column, "")) for column in SUBMISSION_COLUMNS}
        for field, labels in EVAL_FIELDS.items():
            value = normalize_value(prediction.get(field, DEFAULT_PREDICTIONS[field]))
            output[field] = value if value in labels else DEFAULT_PREDICTIONS[field]
        rows.append(output)
    return rows


def write_submission_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUBMISSION_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_validation_metrics_json(
    scores: dict[str, Any] | None,
    output_path: Path,
    *,
    epoch: int | None = None,
    validation_rows: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_epoch": epoch,
        "validation_rows": validation_rows,
        "weighted_score": None,
        "task_f1": {},
    }
    if scores:
        metrics["weighted_score"] = scores.get("final_weighted_score")
        metrics["task_f1"] = {
            field: scores[field]
            for field in TASK_REPORT_ORDER
            if field in scores
        }
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(device) -> None:
    if not uses_cuda(device):
        return

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def format_torch_cuda_report() -> str:
    import torch

    lines = [
        f"PyTorch version: {torch.__version__}",
        f"PyTorch CUDA build: {torch.version.cuda}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA device count: {torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            lines.append(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")
    return "\n".join(lines)


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "CUDA is not available in this PyTorch environment; using CPU. "
            "Please confirm the active Conda environment has a CUDA-enabled PyTorch build."
        )
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch environment cannot use CUDA.\n"
            f"{format_torch_cuda_report()}\n"
            "Install a CUDA-enabled PyTorch wheel, then run again with `--device cuda`."
        )
    return torch.device(device_name)


def uses_cuda(device) -> bool:
    return getattr(device, "type", str(device)) == "cuda"


def make_grad_scaler(device, use_mixed_precision: bool):
    if not (use_mixed_precision and uses_cuda(device)):
        return None

    import torch

    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def autocast_context(device, use_mixed_precision: bool):
    if not (use_mixed_precision and uses_cuda(device)):
        return nullcontext()

    import torch

    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def format_task_name(field: str) -> str:
    return TASK_DISPLAY_NAMES[field]


def format_score_report(
    title: str,
    scores: dict[str, Any] | None,
    epoch: int | None = None,
) -> str:
    if not scores:
        return f"{title}: unavailable"

    lines = [title]
    if epoch:
        lines.append(f"  epoch: {epoch}")
    if "final_weighted_score" in scores:
        lines.append(f"  weighted_score: {float(scores['final_weighted_score']):.4f}")
    for field in TASK_REPORT_ORDER:
        if field in scores:
            lines.append(f"  {field}: {float(scores[field]):.4f}")
    return "\n".join(lines)


def print_score_report(
    title: str,
    scores: dict[str, Any] | None,
    epoch: int | None = None,
) -> None:
    print(format_score_report(title, scores, epoch=epoch))


def format_prediction_distribution(
    rows: list[dict[str, Any]],
    title: str = "Submission prediction distribution",
) -> str:
    lines = [title]
    for field, labels in EVAL_FIELDS.items():
        counts = Counter(normalize_value(row.get(field, "")) for row in rows)
        visible = [f"{label}={counts[label]}" for label in labels if counts[label] > 0]
        lines.append(f"  {field}: {', '.join(visible) if visible else 'none'}")
    return "\n".join(lines)


def print_prediction_distribution(rows: list[dict[str, Any]]) -> None:
    print(format_prediction_distribution(rows))


def require_training_labels(rows: list[dict[str, Any]]) -> None:
    missing = []
    for index, row in enumerate(rows):
        for field, labels in EVAL_FIELDS.items():
            if row.get(field) not in labels:
                missing.append((index, field, row.get(field)))
    if missing:
        preview = ", ".join(
            f"row {idx} field {field}={value!r}" for idx, field, value in missing[:5]
        )
        raise ValueError(f"Training data contains missing or unknown labels: {preview}")


def build_class_weights(
    rows: list[dict[str, Any]],
    field: str,
    label2id: dict[str, int],
    mode: str = "balanced",
    max_weight: float | None = None,
) -> list[float]:
    if mode == "none":
        return [1.0] * len(label2id)
    if mode not in {"balanced", "sqrt"}:
        raise ValueError(f"Unknown class weight mode: {mode}")

    counts = Counter(row[field] for row in rows)
    total = sum(counts.values())
    label_count = len(label2id)
    weights = [0.0] * label_count
    for label, index in label2id.items():
        count = counts.get(label, 0)
        weight = total / (label_count * count) if count else 0.0
        if mode == "sqrt" and weight > 0:
            weight = math.sqrt(weight)
        if max_weight is not None and max_weight > 0:
            weight = min(weight, max_weight)
        weights[index] = weight
    return weights


def build_stratify_labels(rows: list[dict[str, Any]]) -> list[str] | None:
    labels = [
        "|".join(normalize_value(row.get(field, "")) for field in EVAL_FIELDS)
        for row in rows
    ]
    counts = Counter(labels)
    if labels and min(counts.values()) >= 2:
        return labels
    return None


def make_dataset_class(max_len: int):
    import torch
    from torch.utils.data import Dataset

    class ESGDataset(Dataset):
        def __init__(self, rows, tokenizer, label2id=None):
            self.rows = rows
            self.tokenizer = tokenizer
            self.label2id = label2id

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            encoded = self.tokenizer(
                row["data"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

            if self.label2id is not None:
                item["labels"] = {
                    field: torch.tensor(self.label2id[field][row[field]], dtype=torch.long)
                    for field in EVAL_FIELDS
                }
            return item

    return ESGDataset


def collate_batch(batch):
    import torch

    output = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }
    if "labels" in batch[0]:
        output["labels"] = {
            field: torch.stack([item["labels"][field] for item in batch])
            for field in EVAL_FIELDS
        }
    return output


def make_model_class(model_name: str, pooling: str):
    import torch
    from transformers import AutoModel

    class MultiTaskClassifier(torch.nn.Module):
        def __init__(
            self,
            num_labels: dict[str, int],
            dropout_rate: float = 0.2,
            head_hidden_size: int = 256,
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(dropout_rate)
            classifiers = {}
            for field, label_count in num_labels.items():
                if head_hidden_size > 0:
                    classifiers[field] = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, head_hidden_size),
                        torch.nn.GELU(),
                        torch.nn.LayerNorm(head_hidden_size),
                        torch.nn.Dropout(dropout_rate),
                        torch.nn.Linear(head_hidden_size, label_count),
                    )
                else:
                    classifiers[field] = torch.nn.Linear(hidden_size, label_count)
            self.classifiers = torch.nn.ModuleDict(classifiers)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
                pooled = (outputs.last_hidden_state * mask).sum(dim=1)
                pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return {
                field: classifier(pooled)
                for field, classifier in self.classifiers.items()
            }

    return MultiTaskClassifier


def make_loss_functions(
    train_rows,
    label2id,
    device,
    class_weight_mode: str,
    max_class_weight: float,
    label_smoothing: float,
):
    import torch

    losses = {}
    for field in EVAL_FIELDS:
        weight = None
        if class_weight_mode != "none":
            weight = torch.tensor(
                build_class_weights(
                    train_rows,
                    field,
                    label2id[field],
                    mode=class_weight_mode,
                    max_weight=max_class_weight,
                ),
                dtype=torch.float,
                device=device,
            )
        losses[field] = torch.nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )
    return losses


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    losses,
    scaler=None,
    use_mixed_precision: bool = False,
) -> float:
    import torch

    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        non_blocking = uses_cuda(device)
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
        labels = {
            field: values.to(device, non_blocking=non_blocking)
            for field, values in batch["labels"].items()
        }

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_mixed_precision):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum(
                losses[field](logits[field], labels[field]) * FIELD_WEIGHTS[field]
                for field in EVAL_FIELDS
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 50 == 0:
            print(f"  step {step}/{len(dataloader)} loss={loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


def predict_batches(
    model,
    dataloader,
    device,
    id2label: dict[str, dict[int, str]],
    use_mixed_precision: bool = False,
):
    model.eval()
    predictions = []

    import torch

    with torch.no_grad():
        for batch in dataloader:
            non_blocking = uses_cuda(device)
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
            with autocast_context(device, use_mixed_precision):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            batch_size = input_ids.size(0)
            for row_index in range(batch_size):
                prediction = {}
                for field in EVAL_FIELDS:
                    label_id = int(logits[field][row_index].argmax().item())
                    prediction[field] = id2label[field][label_id]
                predictions.append(apply_prediction_constraints(prediction))

    return predictions


def evaluate_predictions(gt_rows, predictions) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    weighted_score = 0.0
    results = {}

    for field, labels in EVAL_FIELDS.items():
        y_true = [row[field] for row in gt_rows]
        y_pred = [row[field] for row in predictions]
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_score += macro_f1 * FIELD_WEIGHTS[field]
        results[field] = macro_f1

    results["final_weighted_score"] = weighted_score
    return results


def train_and_predict(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    model_name: str,
    model_path: Path,
    max_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    dropout_rate: float,
    head_hidden_size: int,
    weight_decay: float,
    warmup_ratio: float,
    early_stopping_patience: int,
    min_delta: float,
    pooling: str,
    class_weight_mode: str,
    max_class_weight: float,
    label_smoothing: float,
    use_mixed_precision: bool,
    metrics_output: Path | None,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    require_training_labels(train_rows)
    if validation_rows:
        require_training_labels(validation_rows)
    seed_everything(seed)
    device = resolve_device(device_name)
    configure_torch_runtime(device)
    label2id, id2label = build_label_maps()
    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ESGDataset = make_dataset_class(max_len)
    MultiTaskClassifier = make_model_class(model_name, pooling)

    train_loader = DataLoader(
        ESGDataset(train_rows, tokenizer, label2id),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        pin_memory=uses_cuda(device),
    )
    valid_loader = None
    if validation_rows:
        valid_loader = DataLoader(
            ESGDataset(validation_rows, tokenizer),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            pin_memory=uses_cuda(device),
        )

    model = MultiTaskClassifier(
        num_labels,
        dropout_rate=dropout_rate,
        head_hidden_size=head_hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    losses = make_loss_functions(
        train_rows,
        label2id,
        device,
        class_weight_mode,
        max_class_weight,
        label_smoothing,
    )
    scaler = make_grad_scaler(device, use_mixed_precision)

    best_score = -1.0
    best_epoch = 0
    best_scores = None
    best_state = None
    epochs_without_improvement = 0
    print(f"Device: {device}")
    print(f"Train rows: {len(train_rows)}; validation rows: {len(validation_rows)}")
    print(
        f"Model: {model_name}; pooling: {pooling}; "
        f"class weights: {class_weight_mode}; mixed precision: "
        f"{use_mixed_precision and uses_cuda(device)}"
    )

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            losses,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
        )
        print(f"  average loss={avg_loss:.4f}")

        if valid_loader is None:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            continue

        valid_predictions = predict_batches(
            model,
            valid_loader,
            device,
            id2label,
            use_mixed_precision=use_mixed_precision,
        )
        scores = evaluate_predictions(validation_rows, valid_predictions)
        score = scores["final_weighted_score"]
        print(f"  validation weighted macro F1={score:.5f}")
        for field in TASK_REPORT_ORDER:
            print(f"    {format_task_name(field)}: {scores[field]:.5f}")

        if score > best_score + min_delta:
            best_score = score
            best_scores = dict(scores)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
            print(f"  saved best checkpoint in memory: {best_score:.5f}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(
                    "  early stopping: validation score did not improve for "
                    f"{early_stopping_patience} epochs"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_scores is not None:
        print_score_report("Best validation result", best_scores, epoch=best_epoch)
    if metrics_output is not None:
        write_validation_metrics_json(
            best_scores,
            metrics_output,
            epoch=best_epoch or None,
            validation_rows=len(validation_rows),
        )
        print(f"Validation metrics written to: {metrics_output}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "max_len": max_len,
            "num_labels": num_labels,
            "dropout_rate": dropout_rate,
            "head_hidden_size": head_hidden_size,
            "pooling": pooling,
            "best_epoch": best_epoch,
            "class_weight_mode": class_weight_mode,
            "max_class_weight": max_class_weight,
            "label_smoothing": label_smoothing,
            "validation_rows": len(validation_rows),
            "best_scores": best_scores,
        },
        model_path,
    )
    print(f"Model checkpoint written to: {model_path}")

    target_loader = DataLoader(
        ESGDataset(target_rows, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=uses_cuda(device),
    )
    return predict_batches(
        model,
        target_loader,
        device,
        id2label,
        use_mixed_precision=use_mixed_precision,
    )


def predict_with_checkpoint(
    target_rows: list[dict[str, Any]],
    *,
    model_path: Path,
    model_name: str,
    max_len: int,
    batch_size: int,
    device_name: str,
    use_mixed_precision: bool,
    metrics_output: Path | None,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device = resolve_device(device_name)
    configure_torch_runtime(device)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name", model_name)
    checkpoint_max_len = int(checkpoint.get("max_len", max_len))
    checkpoint_dropout_rate = float(checkpoint.get("dropout_rate", 0.2))
    checkpoint_head_hidden_size = int(checkpoint.get("head_hidden_size", 0))
    checkpoint_pooling = checkpoint.get("pooling", "cls")
    checkpoint_scores = checkpoint.get("best_scores")
    checkpoint_epoch = checkpoint.get("best_epoch")
    print_score_report(
        "Checkpoint validation result",
        checkpoint_scores,
        epoch=checkpoint_epoch,
    )
    if metrics_output is not None:
        write_validation_metrics_json(
            checkpoint_scores,
            metrics_output,
            epoch=checkpoint_epoch,
            validation_rows=int(checkpoint.get("validation_rows", 0) or 0),
        )
        print(f"Validation metrics written to: {metrics_output}")

    _, id2label = build_label_maps()
    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)
    ESGDataset = make_dataset_class(checkpoint_max_len)
    MultiTaskClassifier = make_model_class(checkpoint_model_name, checkpoint_pooling)
    model = MultiTaskClassifier(
        num_labels,
        dropout_rate=checkpoint_dropout_rate,
        head_hidden_size=checkpoint_head_hidden_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataloader = DataLoader(
        ESGDataset(target_rows, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=uses_cuda(device),
    )
    return predict_batches(
        model,
        dataloader,
        device,
        id2label,
        use_mixed_precision=use_mixed_precision,
    )


def default_target_path(project_root: Path = PROJECT_ROOT) -> Path:
    candidates = [
        project_root / "data" / "vpesg4k_test_2000.json",
        project_root / "data" / "vpesg4k_test_2000.csv",
        project_root / "vpesg4k_test_2000.json",
        project_root / "vpesg4k_test_2000.csv",
        project_root / "data" / "vpesg4k_val_1000.json",
        project_root / "data" / "vpesg4k_val_1000.csv",
        project_root / "vpesg4k_val_1000.json",
        project_root / "vpesg4k_val_1000.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_validation_path(project_root: Path = PROJECT_ROOT) -> Path | None:
    candidates = [
        project_root / "data" / "vpesg4k_val_1000.json",
        project_root / "data" / "vpesg4k_val_1000.csv",
        project_root / "vpesg4k_val_1000.json",
        project_root / "vpesg4k_val_1000.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an AI CUP VeriPromiseESG submission CSV."
    )
    parser.add_argument("--target", type=Path, default=default_target_path())
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "submission.csv")
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "outputs" / "validation_metrics.json")
    parser.add_argument("--train-data", type=Path, default=PROJECT_ROOT / "data" / "vpesg4k_train_1000.json")
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=default_validation_path(),
        help="Labeled validation data used only for scoring and early stopping.",
    )
    parser.add_argument(
        "--no-validation-data",
        action="store_true",
        help="Train without validation scoring or early stopping.",
    )
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "ours_4090.pt")
    parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--head-hidden-size", type=int, default=512)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean"])
    parser.add_argument("--class-weight-mode", default="sqrt", choices=["balanced", "sqrt", "none"])
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--no-download-train",
        action="store_true",
        help="Do not download training data automatically when the checkpoint does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_rows = load_rows(args.target)
    class_weight_mode = "none" if args.no_class_weights else args.class_weight_mode
    use_mixed_precision = not args.no_mixed_precision

    if args.model_path.exists() and not args.force_train:
        print(f"Found model checkpoint: {args.model_path}")
        predictions = predict_with_checkpoint(
            target_rows,
            model_path=args.model_path,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
            device_name=args.device,
            use_mixed_precision=use_mixed_precision,
            metrics_output=args.metrics_output,
        )
    else:
        if args.model_path.exists():
            print(f"Ignoring existing model checkpoint because --force-train was set: {args.model_path}")
        else:
            print(f"Model checkpoint not found: {args.model_path}")
        print("Training a new model, then generating the submission CSV.")
        if not args.no_download_train:
            download_train_data(args.train_data)
        train_rows = load_rows(args.train_data)
        validation_rows = []
        if not args.no_validation_data and args.validation_data is not None:
            validation_rows = load_rows(args.validation_data)
            print(f"Validation data: {args.validation_data}")
        elif args.no_validation_data:
            print("Validation data disabled; training will run for all epochs.")
        else:
            print("Validation data not found; training will run for all epochs.")
        predictions = train_and_predict(
            train_rows,
            validation_rows,
            target_rows,
            model_name=args.model_name,
            model_path=args.model_path,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device_name=args.device,
            dropout_rate=args.dropout_rate,
            head_hidden_size=args.head_hidden_size,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            pooling=args.pooling,
            class_weight_mode=class_weight_mode,
            max_class_weight=args.max_class_weight,
            label_smoothing=args.label_smoothing,
            use_mixed_precision=use_mixed_precision,
            metrics_output=args.metrics_output,
        )

    submission_rows = build_submission_rows(target_rows, predictions)
    write_submission_csv(submission_rows, args.output)
    print(f"Submission CSV written to: {args.output}")
    print(f"Rows: {len(submission_rows)}")
    print_prediction_distribution(submission_rows)


if __name__ == "__main__":
    main()
