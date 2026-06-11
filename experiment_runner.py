from __future__ import annotations

import argparse
import csv
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FINAL_EPOCHS = 8

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
            dropout_rate: float = 0.1,
            head_hidden_size: int = 0,
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
            loss = sum(losses[field](logits[field], labels[field]) for field in EVAL_FIELDS)

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
    apply_constraints: bool = False,
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
                if apply_constraints:
                    prediction = apply_prediction_constraints(prediction)
                predictions.append(prediction)

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
    apply_constraints: bool,
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
            apply_constraints=apply_constraints,
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
        apply_constraints=apply_constraints,
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
    apply_constraints: bool,
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
    checkpoint_dropout_rate = float(checkpoint.get("dropout_rate", 0.1))
    checkpoint_head_hidden_size = int(checkpoint.get("head_hidden_size", 0))
    checkpoint_pooling = checkpoint.get("pooling", "cls")
    checkpoint_scores = checkpoint.get("best_scores")
    checkpoint_epoch = checkpoint.get("best_epoch")
    print_score_report(
        "Checkpoint validation result",
        checkpoint_scores,
        epoch=checkpoint_epoch,
    )
    if checkpoint.get("final_training_mode") == "train_val_merge":
        print(
            "Checkpoint final training"
            f": mode=train+val merge; rows={checkpoint.get('final_train_rows', '?')};"
            f" epochs={checkpoint.get('final_epochs', '?')}"
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
        apply_constraints=apply_constraints,
    )


def load_checkpoint_validation_metadata(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        return {}

    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        print(f"Validation metadata unavailable from checkpoint ({type(exc).__name__}).")
        return {}
    metadata = {}
    for key in ("best_epoch", "best_scores", "validation_rows"):
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    return metadata


def restore_checkpoint_submission_metadata(
    model_path: Path,
    validation_metadata: dict[str, Any],
    *,
    final_train_rows: int,
    final_epochs: int,
) -> None:
    if not model_path.exists():
        return

    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        print(f"Could not restore validation metadata to checkpoint ({type(exc).__name__}).")
        return
    checkpoint.update(validation_metadata)
    checkpoint["final_training_mode"] = "train_val_merge"
    checkpoint["final_train_rows"] = final_train_rows
    checkpoint["final_epochs"] = final_epochs
    torch.save(checkpoint, model_path)


def choose_final_epochs(
    validation_metadata: dict[str, Any],
    *,
    requested_final_epochs: int | None,
    fallback_epochs: int,
) -> int:
    if requested_final_epochs is not None:
        return max(1, requested_final_epochs)

    best_epoch = validation_metadata.get("best_epoch")
    if best_epoch:
        return max(1, int(best_epoch))
    return max(1, fallback_epochs)


def checkpoint_is_current_submission_model(
    model_path: Path,
    *,
    require_merge: bool,
    final_epochs: int | None,
    model_name: str,
    max_len: int,
    pooling: str,
    class_weight_mode: str,
    max_class_weight: float,
    label_smoothing: float,
    dropout_rate: float,
    head_hidden_size: int,
) -> bool:
    if not model_path.exists():
        return False

    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        print(f"Existing checkpoint is not readable; retraining ({type(exc).__name__}).")
        return False

    if require_merge:
        is_current_merge = checkpoint.get("final_training_mode") == "train_val_merge"
        is_legacy_merge = (
            checkpoint.get("train_rows")
            and not checkpoint.get("validation_rows")
            and checkpoint.get("epochs")
        )
        if not (is_current_merge or is_legacy_merge):
            return False

        checkpoint_epochs = checkpoint.get("final_epochs", checkpoint.get("epochs"))
        if final_epochs is not None and int(checkpoint_epochs or 0) != int(final_epochs):
            return False

    expected_values = {
        "model_name": model_name,
        "max_len": max_len,
        "pooling": pooling,
        "class_weight_mode": class_weight_mode,
        "max_class_weight": max_class_weight,
        "label_smoothing": label_smoothing,
        "dropout_rate": dropout_rate,
        "head_hidden_size": head_hidden_size,
    }
    for key, expected in expected_values.items():
        if key in checkpoint and checkpoint[key] != expected:
            return False
    return True


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


import statistics
import subprocess
import sys
from dataclasses import dataclass, fields
from datetime import datetime

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
        str(Path(__file__).resolve()),
        "--mode",
        "single",
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


def run_single(args: argparse.Namespace) -> None:
    target_path = args.target or args.target_data
    target_rows = load_rows(target_path)
    class_weight_mode = "none" if args.no_class_weights else args.class_weight_mode
    use_mixed_precision = not args.no_mixed_precision

    can_use_checkpoint = False
    if args.model_path.exists() and not args.force_train:
        can_use_checkpoint = checkpoint_is_current_submission_model(
            args.model_path,
            require_merge=args.merge_train_val_for_submission,
            final_epochs=args.final_epochs,
            model_name=args.model_name,
            max_len=args.max_len,
            pooling=args.pooling,
            class_weight_mode=class_weight_mode,
            max_class_weight=args.max_class_weight,
            label_smoothing=args.label_smoothing,
            dropout_rate=args.dropout_rate,
            head_hidden_size=args.head_hidden_size,
        )

    if can_use_checkpoint:
        print(f"Found model checkpoint: {args.model_path}")
        predictions = predict_with_checkpoint(
            target_rows,
            model_path=args.model_path,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
            device_name=args.device,
            use_mixed_precision=use_mixed_precision,
            apply_constraints=args.apply_prediction_constraints,
            metrics_output=args.metrics_output,
        )
    else:
        if args.model_path.exists():
            if args.force_train:
                print(f"Ignoring existing model checkpoint because --force-train was set: {args.model_path}")
            else:
                print(
                    "Existing model checkpoint does not match the current merge "
                    f"submission settings; retraining: {args.model_path}"
                )
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
            apply_constraints=args.apply_prediction_constraints,
            metrics_output=args.metrics_output,
        )
        if args.merge_train_val_for_submission and validation_rows:
            validation_metadata = load_checkpoint_validation_metadata(args.model_path)
            final_epochs = choose_final_epochs(
                validation_metadata,
                requested_final_epochs=args.final_epochs,
                fallback_epochs=args.epochs,
            )
            merged_rows = train_rows + validation_rows
            print(
                "Final submission training: merging "
                f"train({len(train_rows)}) + validation({len(validation_rows)}) "
                f"= {len(merged_rows)} rows for {final_epochs} epochs."
            )
            predictions = train_and_predict(
                merged_rows,
                [],
                target_rows,
                model_name=args.model_name,
                model_path=args.model_path,
                max_len=args.max_len,
                batch_size=args.batch_size,
                epochs=final_epochs,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device_name=args.device,
                dropout_rate=args.dropout_rate,
                head_hidden_size=args.head_hidden_size,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                early_stopping_patience=0,
                min_delta=args.min_delta,
                pooling=args.pooling,
                class_weight_mode=class_weight_mode,
                max_class_weight=args.max_class_weight,
                label_smoothing=args.label_smoothing,
                use_mixed_precision=use_mixed_precision,
                apply_constraints=args.apply_prediction_constraints,
                metrics_output=None,
            )
            restore_checkpoint_submission_metadata(
                args.model_path,
                validation_metadata,
                final_train_rows=len(merged_rows),
                final_epochs=final_epochs,
            )
        elif args.merge_train_val_for_submission:
            print("Final train+validation merge skipped because validation data is unavailable.")

    submission_rows = build_submission_rows(target_rows, predictions)
    write_submission_csv(submission_rows, args.output)
    print(f"Submission CSV written to: {args.output}")
    print(f"Rows: {len(submission_rows)}")
    print_prediction_distribution(submission_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run many reproducible VeriPromiseESG experiments, rank by validation "
            "stability, then create final train+val submissions."
        )
    )
    parser.add_argument("--mode", choices=["search", "final", "all", "single"], default="search")
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
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "submission.csv")
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "outputs" / "validation_metrics.json")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "ours.pt")
    parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--head-hidden-size", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    parser.add_argument("--class-weight-mode", default="balanced", choices=["balanced", "sqrt", "none"])
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--apply-prediction-constraints",
        dest="apply_prediction_constraints",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-apply-prediction-constraints",
        dest="apply_prediction_constraints",
        action="store_false",
    )
    parser.add_argument(
        "--merge-train-val-for-submission",
        dest="merge_train_val_for_submission",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-merge-train-val-for-submission",
        dest="merge_train_val_for_submission",
        action="store_false",
    )
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS)
    parser.add_argument("--no-validation-data", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    if args.mode == "single":
        run_single(args)
        return

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
