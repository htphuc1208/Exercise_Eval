from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_DATASET_DIRNAME = "action_dataset"
DEFAULT_SEQUENCE_LENGTH = 64
DEFAULT_AUTO_SEGMENTS_DIRNAME = "segments"
DEFAULT_MANUAL_SEGMENTS_DIRNAME = "manual_segments"

SELECTED_LANDMARKS: tuple[tuple[int, str], ...] = (
    (0, "nose"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
)

SELECTED_LANDMARK_INDICES = [item[0] for item in SELECTED_LANDMARKS]
SELECTED_LANDMARK_NAMES = [item[1] for item in SELECTED_LANDMARKS]
SELECTED_INDEX_BY_LANDMARK = {
    landmark_idx: selected_idx for selected_idx, (landmark_idx, _) in enumerate(SELECTED_LANDMARKS)
}

_BONE_CONNECTIONS_SOURCE: tuple[tuple[int, int], ...] = (
    (0, 11),
    (0, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)

BONE_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (SELECTED_INDEX_BY_LANDMARK[src], SELECTED_INDEX_BY_LANDMARK[dst])
    for src, dst in _BONE_CONNECTIONS_SOURCE
)

SUMMARY_SOURCE_COLUMNS: tuple[str, ...] = (
    "center_y",
    "scale",
    "nose_y_norm",
    "shoulder_center_x_norm",
    "shoulder_center_y_norm",
    "left_wrist_above_left_shoulder",
    "right_wrist_above_right_shoulder",
    "left_wrist_above_head",
    "right_wrist_above_head",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "dist_wrist_wrist",
    "dist_left_wrist_right_ankle",
    "dist_right_wrist_left_ankle",
    "dist_left_wrist_left_hip",
    "dist_right_wrist_right_hip",
    "left_visibility",
    "right_visibility",
    "mean_knee_angle",
    "center_y_velocity",
    "left_wrist_speed",
    "right_wrist_speed",
    "wrist_speed_mean",
    "theta_left",
    "theta_right",
    "dtheta_left",
    "dtheta_right",
    "circle_range_left",
    "circle_range_right",
    "theta_left_xy",
    "theta_right_xy",
    "theta_left_xz",
    "theta_right_xz",
    "circle_range_left_xy",
    "circle_range_right_xy",
    "circle_range_left_xz",
    "circle_range_right_xz",
    "circle_range_left_best",
    "circle_range_right_best",
    "arm_sync",
    "center_y_delta",
    "cross_touch_min",
    "action_1_score",
    "action_2_score",
    "action_3_score",
    "action_4_score",
    "action_5_score",
    "action_2_rescue_score",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_segment_csv_path(video_id: str, directory: Path) -> Path:
    return directory / f"{video_id}_segments.csv"


def ensure_segment_dirs(output_dir: Path) -> tuple[Path, Path]:
    auto_dir = ensure_dir(output_dir / DEFAULT_AUTO_SEGMENTS_DIRNAME)
    manual_dir = ensure_dir(output_dir / DEFAULT_MANUAL_SEGMENTS_DIRNAME)
    return auto_dir, manual_dir


def resolve_segment_csv_path(video_id: str, output_dir: Path) -> Path:
    auto_dir, manual_dir = ensure_segment_dirs(output_dir)
    manual_path = build_segment_csv_path(video_id, manual_dir)
    if manual_path.exists():
        return manual_path
    return build_segment_csv_path(video_id, auto_dir)


def collect_segment_csv_paths(output_dir: Path) -> list[Path]:
    auto_dir, manual_dir = ensure_segment_dirs(output_dir)
    by_video_id: dict[str, Path] = {}
    for path in sorted(auto_dir.glob("*_segments.csv")):
        video_id = path.stem.replace("_segments", "")
        by_video_id[video_id] = path
    for path in sorted(manual_dir.glob("*_segments.csv")):
        video_id = path.stem.replace("_segments", "")
        by_video_id[video_id] = path
    return [by_video_id[video_id] for video_id in sorted(by_video_id)]


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def parse_boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def random_video_split(
    video_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative.")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    videos = list(dict.fromkeys(video_ids))
    if not videos:
        return {}

    rng = np.random.default_rng(seed)
    rng.shuffle(videos)

    total = len(videos)
    raw_counts = ratios * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - counts.sum()
    order = np.argsort(raw_counts - counts)[::-1]
    for idx in order[:remainder]:
        counts[idx] += 1

    if total >= 3:
        if counts[1] == 0:
            counts[1] = 1
            counts[0] = max(1, counts[0] - 1)
        if counts[2] == 0:
            counts[2] = 1
            counts[0] = max(1, counts[0] - 1)

    while counts.sum() > total:
        largest = int(np.argmax(counts))
        if counts[largest] > 1:
            counts[largest] -= 1
        else:
            break

    while counts.sum() < total:
        counts[0] += 1

    boundaries = np.cumsum(counts)
    split_labels = ("train", "val", "test")
    split_map: dict[str, str] = {}
    start = 0
    for split_idx, end in enumerate(boundaries):
        for video_id in videos[start:end]:
            split_map[video_id] = split_labels[split_idx]
        start = end
    return split_map


def resample_temporal_array(array: np.ndarray, target_length: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if arr.ndim == 0:
        raise ValueError("array must have at least one dimension")
    if arr.shape[0] == target_length:
        return arr.copy()

    output_shape = (target_length,) + arr.shape[1:]
    if arr.shape[0] == 0:
        return np.full(output_shape, np.nan, dtype=np.float32)

    if arr.shape[0] == 1:
        return np.repeat(arr.astype(np.float32), target_length, axis=0)

    source_x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)

    flat = arr.reshape(arr.shape[0], -1)
    out = np.full((target_length, flat.shape[1]), np.nan, dtype=np.float32)
    for col_idx in range(flat.shape[1]):
        series = flat[:, col_idx]
        valid = np.isfinite(series)
        if not np.any(valid):
            continue
        if valid.sum() == 1:
            out[:, col_idx] = series[valid][0]
            continue
        out[:, col_idx] = np.interp(target_x, source_x[valid], series[valid]).astype(np.float32)
    return out.reshape(output_shape)


def compute_bone_vectors(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    bones = np.full((joints.shape[0], len(BONE_CONNECTIONS), 3), np.nan, dtype=np.float32)
    for bone_idx, (src_idx, dst_idx) in enumerate(BONE_CONNECTIONS):
        bones[:, bone_idx] = joints[:, dst_idx] - joints[:, src_idx]
    return bones


def compute_motion(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[0] <= 1:
        return out
    out[1:] = arr[1:] - arr[:-1]
    return out


def nan_summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "range": math.nan,
        }
    valid_values = arr[valid]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    return {
        "mean": float(valid_values.mean()),
        "std": float(valid_values.std()),
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
    }


def count_signal_peaks(values: np.ndarray, threshold: float | None = None) -> int:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return 0
    valid = np.isfinite(arr)
    peak_count = 0
    for idx in range(1, arr.size - 1):
        if not (valid[idx - 1] and valid[idx] and valid[idx + 1]):
            continue
        if threshold is not None and arr[idx] < threshold:
            continue
        if arr[idx] >= arr[idx - 1] and arr[idx] >= arr[idx + 1] and (arr[idx] > arr[idx - 1] or arr[idx] > arr[idx + 1]):
            peak_count += 1
    return peak_count


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if true.size == 0:
        return {
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    pred = (prob >= threshold).astype(np.int8)
    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float(np.mean(pred == true))
    return {
        "count": int(true.size),
        "positive_count": int(np.sum(true == 1)),
        "negative_count": int(np.sum(true == 0)),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_video_exact_match(predictions_df: pd.DataFrame, split_name: str) -> dict[str, object]:
    subset = predictions_df[predictions_df["split"] == split_name].copy()
    if subset.empty:
        return {
            "split": split_name,
            "video_count": 0,
            "complete_video_count": 0,
            "exact_match": math.nan,
            "rows": [],
        }

    rows: list[dict[str, object]] = []
    complete = 0
    correct = 0
    for video_id, group in subset.groupby("video_id", sort=True):
        action_group = group.sort_values("action_id")
        if action_group["action_id"].nunique() != 5:
            continue
        complete += 1
        true_bits = "".join(str(int(value)) for value in action_group["y_true"].to_numpy(dtype=np.int8))
        pred_bits = "".join(str(int(value)) for value in action_group["y_pred"].to_numpy(dtype=np.int8))
        is_exact = int(true_bits == pred_bits)
        correct += is_exact
        rows.append(
            {
                "video_id": video_id,
                "split": split_name,
                "true_bits": true_bits,
                "pred_bits": pred_bits,
                "exact_match": is_exact,
            }
        )

    metric = correct / complete if complete > 0 else math.nan
    return {
        "split": split_name,
        "video_count": int(subset["video_id"].nunique()),
        "complete_video_count": complete,
        "exact_match": metric,
        "rows": rows,
    }


def load_manifest_feature_columns(manifest_df: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in manifest_df.columns
        if column.startswith("feat_") or column.startswith("rep_")
    )
