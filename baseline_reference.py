from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
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
        predictions.append(prediction)
    return predictions


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


def format_task_name(field: str) -> str:
    return TASK_DISPLAY_NAMES[field]


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


def make_model_class(model_name: str):
    import torch
    from transformers import AutoModel

    class MultiTaskClassifier(torch.nn.Module):
        def __init__(self, num_labels: dict[str, int], dropout_rate: float = 0.1):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(dropout_rate)
            self.classifiers = torch.nn.ModuleDict(
                {
                    field: torch.nn.Linear(hidden_size, label_count)
                    for field, label_count in num_labels.items()
                }
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return {
                field: classifier(pooled)
                for field, classifier in self.classifiers.items()
            }

    return MultiTaskClassifier


def train_one_epoch(model, dataloader, optimizer, scheduler, device) -> float:
    import torch

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        non_blocking = uses_cuda(device)
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
        labels = {
            field: values.to(device, non_blocking=non_blocking)
            for field, values in batch["labels"].items()
        }

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = sum(criterion(logits[field], labels[field]) for field in EVAL_FIELDS)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 50 == 0:
            print(f"  step {step}/{len(dataloader)} loss={loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


def predict_batches(model, dataloader, device, id2label: dict[str, dict[int, str]]):
    model.eval()
    predictions = []

    import torch

    with torch.no_grad():
        for batch in dataloader:
            non_blocking = uses_cuda(device)
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            batch_size = input_ids.size(0)
            for row_index in range(batch_size):
                prediction = {}
                for field in EVAL_FIELDS:
                    label_id = int(logits[field][row_index].argmax().item())
                    prediction[field] = id2label[field][label_id]
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
    target_rows: list[dict[str, Any]],
    *,
    model_name: str,
    model_path: Path,
    max_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    validation_size: float,
    seed: int,
    device_name: str,
):
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    require_training_labels(train_rows)
    seed_everything(seed)
    device = resolve_device(device_name)
    label2id, id2label = build_label_maps()
    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}

    if validation_size > 0:
        train_split, valid_split = train_test_split(
            train_rows,
            test_size=validation_size,
            random_state=seed,
        )
    else:
        train_split, valid_split = train_rows, []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ESGDataset = make_dataset_class(max_len)
    MultiTaskClassifier = make_model_class(model_name)

    train_loader = DataLoader(
        ESGDataset(train_split, tokenizer, label2id),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        pin_memory=uses_cuda(device),
    )
    valid_loader = None
    if valid_split:
        valid_loader = DataLoader(
            ESGDataset(valid_split, tokenizer),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            pin_memory=uses_cuda(device),
        )

    model = MultiTaskClassifier(num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_score = -1.0
    best_state = None
    print(f"Device: {device}")
    print(f"Train rows: {len(train_split)}; validation rows: {len(valid_split)}")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  average loss={avg_loss:.4f}")

        if valid_loader is None:
            best_state = model.state_dict()
            continue

        valid_predictions = predict_batches(model, valid_loader, device, id2label)
        scores = evaluate_predictions(valid_split, valid_predictions)
        score = scores["final_weighted_score"]
        print(f"  validation weighted macro F1={score:.5f}")
        for field in TASK_REPORT_ORDER:
            print(f"    {format_task_name(field)}: {scores[field]:.5f}")

        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            print(f"  saved best checkpoint in memory: {best_score:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "max_len": max_len,
            "num_labels": num_labels,
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
    return predict_batches(model, target_loader, device, id2label)


def predict_with_checkpoint(
    target_rows: list[dict[str, Any]],
    *,
    model_path: Path,
    model_name: str,
    max_len: int,
    batch_size: int,
    device_name: str,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device = resolve_device(device_name)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name", model_name)
    checkpoint_max_len = int(checkpoint.get("max_len", max_len))

    _, id2label = build_label_maps()
    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)
    ESGDataset = make_dataset_class(checkpoint_max_len)
    MultiTaskClassifier = make_model_class(checkpoint_model_name)
    model = MultiTaskClassifier(num_labels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataloader = DataLoader(
        ESGDataset(target_rows, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=uses_cuda(device),
    )
    return predict_batches(model, dataloader, device, id2label)


def default_target_path(project_root: Path = PROJECT_ROOT) -> Path:
    candidates = [
        project_root / "data" / "vpesg4k_val_1000.json",
        project_root / "data" / "vpesg4k_val_1000.csv",
        project_root / "vpesg4k_val_1000.json",
        project_root / "vpesg4k_val_1000.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an AI CUP VeriPromiseESG submission CSV."
    )
    parser.add_argument("--target", type=Path, default=default_target_path())
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "submission.csv")
    parser.add_argument("--train-data", type=Path, default=PROJECT_ROOT / "data" / "vpesg4k_train_1000.json")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "baseline_reference.pt")
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-size", type=float, default=0.2)
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

    if args.model_path.exists():
        print(f"Found model checkpoint: {args.model_path}")
        predictions = predict_with_checkpoint(
            target_rows,
            model_path=args.model_path,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    else:
        print(f"Model checkpoint not found: {args.model_path}")
        print("Training a new model, then generating the submission CSV.")
        if not args.no_download_train:
            download_train_data(args.train_data)
        train_rows = load_rows(args.train_data)
        predictions = train_and_predict(
            train_rows,
            target_rows,
            model_name=args.model_name,
            model_path=args.model_path,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validation_size=args.validation_size,
            seed=args.seed,
            device_name=args.device,
        )

    submission_rows = build_submission_rows(target_rows, predictions)
    write_submission_csv(submission_rows, args.output)
    print(f"Submission CSV written to: {args.output}")
    print(f"Rows: {len(submission_rows)}")


if __name__ == "__main__":
    main()
