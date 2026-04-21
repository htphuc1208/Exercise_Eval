from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from action_training_common import (
    DEFAULT_DATASET_DIRNAME,
    DEFAULT_OUTPUT_DIR,
    compute_binary_metrics,
    compute_video_exact_match,
    load_manifest_feature_columns,
    write_json,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shared small temporal model on action-level sequence tensors."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_DATASET_DIRNAME,
        help="Directory containing action_dataset_manifest.csv and sequence tensors.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to action_dataset_manifest.csv. Default: <dataset-dir>/action_dataset_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "training_deep",
        help="Directory for model checkpoints and evaluation artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs. Default: 40")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size. Default: 16")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Temporal hidden width. Default: 128")
    parser.add_argument("--summary-dim", type=int, default=64, help="Summary branch hidden dim. Default: 64")
    parser.add_argument("--action-embed-dim", type=int, default=16, help="Action embedding dim. Default: 16")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default: 0.2")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate. Default: 1e-3")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay. Default: 1e-4")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience. Default: 8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device, e.g. cpu, cuda, cuda:0. Default: auto",
    )
    return parser.parse_args()


class ArrayDataset(Dataset):
    def __init__(
        self,
        sequence_array: "torch.Tensor",
        summary_array: "torch.Tensor",
        action_array: "torch.Tensor",
        label_array: "torch.Tensor",
    ) -> None:
        self.sequence_array = sequence_array
        self.summary_array = summary_array
        self.action_array = action_array
        self.label_array = label_array

    def __len__(self) -> int:
        return int(self.label_array.shape[0])

    def __getitem__(self, index: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        return (
            self.sequence_array[index],
            self.summary_array[index],
            self.action_array[index],
            self.label_array[index],
        )


if torch is not None and nn is not None:
    class TemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            padding = dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
            self.norm1 = nn.BatchNorm1d(channels)
            self.norm2 = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            residual = inputs
            outputs = self.conv1(inputs)
            outputs = self.norm1(outputs)
            outputs = torch.relu(outputs)
            outputs = self.dropout(outputs)
            outputs = self.conv2(outputs)
            outputs = self.norm2(outputs)
            outputs = torch.relu(outputs + residual)
            return self.dropout(outputs)


    class SharedSequenceClassifier(nn.Module):
        def __init__(
            self,
            sequence_dim: int,
            summary_dim: int,
            hidden_dim: int,
            summary_hidden_dim: int,
            action_embed_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Conv1d(sequence_dim, hidden_dim, kernel_size=1)
            self.temporal_blocks = nn.ModuleList(
                [
                    TemporalBlock(hidden_dim, dilation=1, dropout=dropout),
                    TemporalBlock(hidden_dim, dilation=2, dropout=dropout),
                    TemporalBlock(hidden_dim, dilation=4, dropout=dropout),
                ]
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.summary_mlp = nn.Sequential(
                nn.Linear(summary_dim, summary_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(summary_hidden_dim, summary_hidden_dim),
                nn.ReLU(),
            )
            self.action_embedding = nn.Embedding(5, action_embed_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + summary_hidden_dim + action_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            sequence_inputs: "torch.Tensor",
            summary_inputs: "torch.Tensor",
            action_inputs: "torch.Tensor",
        ) -> "torch.Tensor":
            outputs = sequence_inputs.transpose(1, 2)
            outputs = self.input_proj(outputs)
            for block in self.temporal_blocks:
                outputs = block(outputs)
            seq_repr = self.pool(outputs).squeeze(-1)
            summary_repr = self.summary_mlp(summary_inputs)
            action_repr = self.action_embedding(action_inputs)
            combined = torch.cat([seq_repr, summary_repr, action_repr], dim=1)
            return self.classifier(combined).squeeze(1)


def determine_device(requested_device: str) -> "torch.device":
    assert torch is not None
    if requested_device != "auto":
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    assert torch is not None
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequence_matrix(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        joint = data["joint"].astype(np.float32)
        visibility = data["visibility"].astype(np.float32)
        bone = data["bone"].astype(np.float32)
        motion = data["motion"].astype(np.float32)
        bone_motion = data["bone_motion"].astype(np.float32)

    sequence_matrix = np.concatenate(
        [
            joint.reshape(joint.shape[0], -1),
            visibility.reshape(visibility.shape[0], -1),
            bone.reshape(bone.shape[0], -1),
            motion.reshape(motion.shape[0], -1),
            bone_motion.reshape(bone_motion.shape[0], -1),
        ],
        axis=1,
    )
    return sequence_matrix.astype(np.float32)


def compute_feature_statistics(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[str]]:
    keep_columns: list[str] = []
    fill_values: dict[str, float] = {}
    feature_means: dict[str, float] = {}
    feature_stds: dict[str, float] = {}

    for feature_name in feature_columns:
        values = pd.to_numeric(train_df[feature_name], errors="coerce").to_numpy(dtype=np.float32)
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            continue
        keep_columns.append(feature_name)
        fill_values[feature_name] = float(np.median(valid))
        feature_means[feature_name] = float(valid.mean())
        std_value = float(valid.std())
        feature_stds[feature_name] = std_value if std_value > 1e-6 else 1.0

    return fill_values, feature_means, feature_stds, keep_columns


def prepare_summary_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
) -> np.ndarray:
    matrix = df.reindex(columns=feature_columns).to_numpy(dtype=np.float32, copy=True)
    for col_idx, feature_name in enumerate(feature_columns):
        column = matrix[:, col_idx]
        invalid = ~np.isfinite(column)
        if np.any(invalid):
            column[invalid] = fill_values[feature_name]
        column = (column - feature_means[feature_name]) / feature_stds[feature_name]
        matrix[:, col_idx] = column
    return matrix.astype(np.float32)


def build_split_arrays(
    split_df: pd.DataFrame,
    dataset_root: Path,
    feature_columns: list[str],
    fill_values: dict[str, float],
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
) -> dict[str, object]:
    if split_df.empty:
        return {
            "df": split_df.copy(),
            "sequence": np.empty((0, 0, 0), dtype=np.float32),
            "summary": np.empty((0, len(feature_columns)), dtype=np.float32),
            "action": np.empty(0, dtype=np.int64),
            "label": np.empty(0, dtype=np.float32),
        }

    sequence_list: list[np.ndarray] = []
    for relpath in split_df["sequence_tensor_path"]:
        npz_path = (dataset_root / str(relpath)).resolve()
        sequence_list.append(load_sequence_matrix(npz_path))

    sequence_array = np.stack(sequence_list, axis=0).astype(np.float32)
    summary_array = prepare_summary_matrix(
        split_df,
        feature_columns=feature_columns,
        fill_values=fill_values,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    action_array = split_df["action_id"].to_numpy(dtype=np.int64) - 1
    label_array = split_df["target_label"].astype(int).to_numpy(dtype=np.float32)
    return {
        "df": split_df.copy(),
        "sequence": sequence_array,
        "summary": summary_array,
        "action": action_array,
        "label": label_array,
    }


def evaluate_predictions(split_df: pd.DataFrame, logits: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_true = split_df["target_label"].astype(int).to_numpy(dtype=np.int8)
    y_pred = (y_prob >= 0.5).astype(np.int8)

    predictions = split_df[
        [
            "sample_id",
            "video_id",
            "action_id",
            "action_name",
            "segment_status",
            "split",
        ]
    ].copy()
    predictions["y_true"] = y_true
    predictions["y_prob"] = y_prob.astype(np.float32)
    predictions["y_pred"] = y_pred.astype(np.int8)

    overall_metrics = compute_binary_metrics(y_true, y_prob)
    action_metrics = {
        f"action_{action_id}": compute_binary_metrics(
            predictions[predictions["action_id"] == action_id]["y_true"].to_numpy(dtype=np.int8),
            predictions[predictions["action_id"] == action_id]["y_prob"].to_numpy(dtype=np.float32),
        )
        for action_id in sorted(predictions["action_id"].unique())
    }
    return predictions, {"overall": overall_metrics, "per_action": action_metrics}


def forward_logits(
    model: SharedSequenceClassifier,
    split_arrays: dict[str, object],
    device: "torch.device",
) -> np.ndarray:
    assert torch is not None
    sequence_array = split_arrays["sequence"]
    if sequence_array.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.from_numpy(sequence_array).to(device)
        summary_tensor = torch.from_numpy(split_arrays["summary"]).to(device)
        action_tensor = torch.from_numpy(split_arrays["action"]).to(device)
        logits = model(sequence_tensor, summary_tensor, action_tensor)
    return logits.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError(f"PyTorch is not installed: {TORCH_IMPORT_ERROR}")
    set_seed(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else (dataset_dir / "action_dataset_manifest.csv").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    usable_df = manifest_df[
        (manifest_df["is_labeled"] == 1)
        & (manifest_df["sequence_ready"] == 1)
        & (manifest_df["split"].isin(["train", "val", "test"]))
    ].copy()
    if usable_df.empty:
        raise SystemExit("Khong co sample nao co label + bounds de train deep model.")

    feature_columns = load_manifest_feature_columns(usable_df)
    fill_values, feature_means, feature_stds, keep_columns = compute_feature_statistics(
        usable_df[usable_df["split"] == "train"].copy(),
        feature_columns=feature_columns,
    )
    if not keep_columns:
        raise SystemExit("Khong co feature summary nao hop le de train deep model.")

    split_arrays = {
        split_name: build_split_arrays(
            split_df=usable_df[usable_df["split"] == split_name].copy(),
            dataset_root=dataset_dir,
            feature_columns=keep_columns,
            fill_values=fill_values,
            feature_means=feature_means,
            feature_stds=feature_stds,
        )
        for split_name in ("train", "val", "test")
    }

    if split_arrays["train"]["sequence"].shape[0] == 0:
        raise SystemExit("Khong co sample train nao cho deep model.")

    sequence_dim = int(split_arrays["train"]["sequence"].shape[2])
    summary_dim = int(split_arrays["train"]["summary"].shape[1])
    device = determine_device(args.device)

    model = SharedSequenceClassifier(
        sequence_dim=sequence_dim,
        summary_dim=summary_dim,
        hidden_dim=args.hidden_dim,
        summary_hidden_dim=args.summary_dim,
        action_embed_dim=args.action_embed_dim,
        dropout=args.dropout,
    ).to(device)

    train_dataset = ArrayDataset(
        sequence_array=torch.from_numpy(split_arrays["train"]["sequence"]),
        summary_array=torch.from_numpy(split_arrays["train"]["summary"]),
        action_array=torch.from_numpy(split_arrays["train"]["action"]),
        label_array=torch.from_numpy(split_arrays["train"]["label"]),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_labels = split_arrays["train"]["label"]
    positive_count = float(np.sum(train_labels == 1.0))
    negative_count = float(np.sum(train_labels == 0.0))
    pos_weight_value = (negative_count / positive_count) if positive_count > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_val_f1 = -1.0
    best_val_loss = math.inf
    patience_counter = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for sequence_batch, summary_batch, action_batch, label_batch in train_loader:
            sequence_batch = sequence_batch.to(device)
            summary_batch = summary_batch.to(device)
            action_batch = action_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(sequence_batch, summary_batch, action_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_logits = forward_logits(model, split_arrays["train"], device)
        train_metrics = compute_binary_metrics(
            split_arrays["train"]["label"].astype(np.int8),
            1.0 / (1.0 + np.exp(-train_logits)),
        )

        val_logits = forward_logits(model, split_arrays["val"], device)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits)) if val_logits.size else np.empty(0, dtype=np.float32)
        val_metrics = compute_binary_metrics(
            split_arrays["val"]["label"].astype(np.int8),
            val_prob,
        )
        if val_logits.size:
            val_targets_tensor = torch.from_numpy(split_arrays["val"]["label"]).to(device)
            val_logits_tensor = torch.from_numpy(val_logits).to(device)
            val_loss = float(criterion(val_logits_tensor, val_targets_tensor).detach().cpu().item())
        else:
            val_loss = math.nan

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
                "train_f1": float(train_metrics["f1"]),
                "val_loss": val_loss,
                "val_f1": float(val_metrics["f1"]) if not math.isnan(val_metrics["f1"]) else math.nan,
            }
        )

        current_val_f1 = float(val_metrics["f1"]) if not math.isnan(val_metrics["f1"]) else -1.0
        improve = (current_val_f1 > best_val_f1) or (
            math.isclose(current_val_f1, best_val_f1) and val_loss < best_val_loss
        )
        if improve:
            best_val_f1 = current_val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"epoch={epoch:02d} train_loss={history[-1]['train_loss']:.4f} "
            f"train_f1={history[-1]['train_f1']:.4f} val_loss={val_loss:.4f} val_f1={current_val_f1:.4f}"
        )

        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state)

    split_metrics_payload: dict[str, object] = {}
    all_predictions: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        logits = forward_logits(model, split_arrays[split_name], device)
        predictions_df, metrics = evaluate_predictions(split_arrays[split_name]["df"], logits)
        all_predictions.append(predictions_df)
        split_metrics_payload[split_name] = metrics

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    predictions_path = output_dir / "deep_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    video_exact_match_rows: list[dict[str, object]] = []
    video_exact_match_payload: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        exact_match = compute_video_exact_match(predictions_df, split_name)
        video_exact_match_payload[split_name] = {
            key: value for key, value in exact_match.items() if key != "rows"
        }
        video_exact_match_rows.extend(exact_match["rows"])
    video_exact_match_path = output_dir / "deep_video_exact_match.csv"
    pd.DataFrame(video_exact_match_rows).to_csv(video_exact_match_path, index=False)

    checkpoint_path = output_dir / "shared_sequence_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_dim": sequence_dim,
            "summary_dim": summary_dim,
            "hidden_dim": args.hidden_dim,
            "summary_hidden_dim": args.summary_dim,
            "action_embed_dim": args.action_embed_dim,
            "dropout": args.dropout,
            "feature_columns": keep_columns,
            "fill_values": fill_values,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    history_path = output_dir / "deep_training_history.json"
    metrics_path = output_dir / "deep_metrics.json"
    write_json(history_path, history)
    write_json(
        metrics_path,
        {
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "best_epoch": best_epoch,
            "sequence_dim": sequence_dim,
            "summary_dim": summary_dim,
            "feature_count": len(keep_columns),
            "split_metrics": split_metrics_payload,
            "video_exact_match": video_exact_match_payload,
        },
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"History: {history_path}")
    print(f"Exact match: {video_exact_match_path}")


if __name__ == "__main__":
    main()
