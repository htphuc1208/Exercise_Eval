from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from action_training_common import DEFAULT_OUTPUT_DIR, ensure_dir, write_json
from test_pose_extract import DEFAULT_VIDEO_DIR, list_videos_in_dir, resolve_input_paths


ACTION_NAMES = {
    1: "raise_lr",
    2: "arm_circle",
    3: "squat",
    4: "cross_touch",
    5: "side_bend",
}

EXPECTED_REP_COUNT = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
}

ACTION_THRESHOLDS = {
    1: {"min_distance": 12, "prominence": 0.10, "height": 0.08, "cycle_recovery": 0.08, "cycle_window": 12},
    3: {"min_distance": 15, "prominence": 8.0, "hip_drop": 0.04, "cycle_recovery": 10.0, "cycle_window": 14},
    4: {"min_distance": 12, "prominence": 0.08, "bend_gate": 0.22, "cycle_recovery": 0.05, "cycle_window": 12},
    5: {"min_distance": 12, "prominence": 0.04, "gate": 0.18, "lateral": 0.12, "cycle_recovery": 0.03, "cycle_window": 12},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count reps and sub-events per segmented action, then export QA/debug artifacts."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help=(
            "Optional video files, directories, or stems. If omitted, all segment files under "
            "<output-dir>/segments are processed."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory used to resolve relative video names. Default: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute rep artifacts even if both rep_events.json and rep_summary.csv already exist.",
    )
    return parser.parse_args()


def ensure_rep_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    frame_features_dir = output_dir / "frame_features"
    segments_dir = output_dir / "segments"
    rep_events_dir = ensure_dir(output_dir / "rep_events")
    rep_summary_dir = ensure_dir(output_dir / "rep_summary")
    return frame_features_dir, segments_dir, rep_events_dir, rep_summary_dir


def build_frame_features_path(video_id: str, frame_features_dir: Path) -> Path:
    return frame_features_dir / f"{video_id}_frame_features.csv"


def build_segments_path(video_id: str, segments_dir: Path) -> Path:
    return segments_dir / f"{video_id}_segments.csv"


def build_rep_events_path(video_id: str, rep_events_dir: Path) -> Path:
    return rep_events_dir / f"{video_id}_rep_events.json"


def build_rep_summary_path(video_id: str, rep_summary_dir: Path) -> Path:
    return rep_summary_dir / f"{video_id}_rep_summary.csv"


def collect_video_ids(args: argparse.Namespace) -> list[str]:
    _, segments_dir, _, _ = ensure_rep_dirs(args.output_dir.resolve())
    if not args.videos:
        return sorted(path.stem.replace("_segments", "") for path in segments_dir.glob("*_segments.csv"))

    video_ids: list[str] = []
    seen: set[str] = set()
    for token in args.videos:
        if token.lower() == "all":
            return sorted(path.stem.replace("_segments", "") for path in segments_dir.glob("*_segments.csv"))

        segment_candidate = segments_dir / token
        if segment_candidate.exists() and segment_candidate.suffix.lower() == ".csv":
            video_id = segment_candidate.stem.replace("_segments", "")
            if video_id not in seen:
                video_ids.append(video_id)
                seen.add(video_id)
            continue

        stem_candidate = segments_dir / f"{token}_segments.csv"
        if stem_candidate.exists():
            if token not in seen:
                video_ids.append(token)
                seen.add(token)
            continue

        try:
            resolved_paths = resolve_input_paths(token, args.video_dir)
        except FileNotFoundError:
            continue

        for resolved_path in resolved_paths:
            video_id = resolved_path.stem
            if video_id not in seen:
                video_ids.append(video_id)
                seen.add(video_id)

    return video_ids


def rep_outputs_ready(video_id: str, output_dir: Path) -> bool:
    _, _, rep_events_dir, rep_summary_dir = ensure_rep_dirs(output_dir)
    return build_rep_events_path(video_id, rep_events_dir).exists() and build_rep_summary_path(video_id, rep_summary_dir).exists()


def clip_strength(values: np.ndarray | float, low: float, high: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    valid = np.isfinite(arr)
    if high <= low:
        return out
    scaled = (arr[valid] - low) / (high - low)
    out[valid] = np.clip(scaled, 0.0, 1.0)
    return out


def descending_strength(values: np.ndarray | float, low: float, high: float) -> np.ndarray:
    return clip_strength(-(np.asarray(values, dtype=np.float32)), -high, -low)


def safe_nanmean(values: list[float]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def safe_nanmin(values: list[float]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(min(finite)) if finite else math.nan


def safe_rowwise_nanmean(stacked: np.ndarray) -> np.ndarray:
    values = np.asarray(stacked, dtype=np.float32)
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    for idx in range(values.shape[0]):
        row = values[idx]
        valid = row[np.isfinite(row)]
        if valid.size:
            out[idx] = float(valid.mean())
    return out


def get_segment_window(
    features_df: pd.DataFrame,
    segment_row: pd.Series,
) -> tuple[pd.DataFrame, int | None, int | None]:
    if pd.isna(segment_row.get("start_frame")) or pd.isna(segment_row.get("end_frame")):
        return pd.DataFrame(columns=features_df.columns), None, None
    start_frame = int(segment_row["start_frame"])
    end_frame = int(segment_row["end_frame"])
    frame_mask = (features_df["frame_idx"] >= start_frame) & (features_df["frame_idx"] <= end_frame)
    return features_df.loc[frame_mask].copy(), start_frame, end_frame


def get_numeric_series(
    frame_window: pd.DataFrame,
    primary_name: str,
    fallback_names: tuple[str, ...] = (),
) -> np.ndarray:
    for column_name in (primary_name, *fallback_names):
        if column_name in frame_window.columns:
            return pd.to_numeric(frame_window[column_name], errors="coerce").to_numpy(dtype=np.float32)
    return np.full(frame_window.shape[0], np.nan, dtype=np.float32)


def detect_extrema(
    signal: np.ndarray,
    *,
    valley: bool,
    distance: int,
    prominence: float,
    height: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size < 3:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    valid_signal = arr[valid_indices]
    search_signal = -valid_signal if valley else valid_signal
    kwargs: dict[str, object] = {
        "distance": max(1, int(distance)),
        "prominence": max(0.0, float(prominence)),
    }
    if height is not None:
        kwargs["height"] = -float(height) if valley else float(height)

    peak_indices, properties = find_peaks(search_signal, **kwargs)
    mapped_indices = valid_indices[peak_indices].astype(np.int32)
    mapped_values = arr[mapped_indices].astype(np.float32)
    mapped_prominences = np.asarray(properties.get("prominences", np.zeros(mapped_indices.size)), dtype=np.float32)
    return mapped_indices, mapped_values, mapped_prominences


def complete_cycle_check(
    signal: np.ndarray,
    index: int,
    *,
    valley: bool,
    min_recovery: float,
    window: int,
) -> bool:
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if index < 0 or index >= arr.size or not np.isfinite(arr[index]):
        return False

    left_slice = arr[max(0, index - window):index]
    right_slice = arr[index + 1:min(arr.size, index + 1 + window)]
    left_valid = left_slice[np.isfinite(left_slice)]
    right_valid = right_slice[np.isfinite(right_slice)]
    if left_valid.size == 0 or right_valid.size == 0:
        return False

    if valley:
        left_recovery = float(left_valid.max() - arr[index])
        right_recovery = float(right_valid.max() - arr[index])
    else:
        left_recovery = float(arr[index] - left_valid.min())
        right_recovery = float(arr[index] - right_valid.min())
    return left_recovery >= min_recovery and right_recovery >= min_recovery


def blank_summary(
    video_id: str,
    action_id: int,
    action_name: str,
    segment_status: str,
    primary_signal_name: str,
    rep_status: str,
) -> dict[str, object]:
    return {
        "video_id": video_id,
        "action_id": action_id,
        "action_name": action_name,
        "segment_status": segment_status,
        "rep_status": rep_status,
        "rep_count": math.nan,
        "expected_count": EXPECTED_REP_COUNT[action_id],
        "rep_count_match": math.nan,
        "event_count_left": 0,
        "event_count_right": 0,
        "quality_mean": math.nan,
        "quality_min": math.nan,
        "primary_signal_name": primary_signal_name,
    }


def analyze_action1(video_id: str, action_name: str, segment_status: str, frame_window: pd.DataFrame, start_frame: int) -> tuple[dict[str, object], dict[str, object]]:
    summary = blank_summary(video_id, 1, action_name, segment_status, "left_wrist_above_head,right_wrist_above_head", "no_signal")
    if frame_window.empty:
        summary["rep_status"] = "no_bounds"
        return summary, {"action_id": 1, "signal_name": "left_wrist_above_head,right_wrist_above_head", "rep_count": None, "expected_count": 2, "event_frames": [], "event_sides": [], "event_quality": [], "extra_metrics": {}}

    cfg = ACTION_THRESHOLDS[1]
    left_signal = pd.to_numeric(frame_window["left_wrist_above_head"], errors="coerce").to_numpy(dtype=np.float32)
    right_signal = pd.to_numeric(frame_window["right_wrist_above_head"], errors="coerce").to_numpy(dtype=np.float32)

    left_idx, left_values, left_prom = detect_extrema(left_signal, valley=False, distance=cfg["min_distance"], prominence=cfg["prominence"], height=cfg["height"])
    right_idx, right_values, right_prom = detect_extrema(right_signal, valley=False, distance=cfg["min_distance"], prominence=cfg["prominence"], height=cfg["height"])

    left_keep: list[int] = []
    left_quality: list[float] = []
    for idx, peak_value, prominence in zip(left_idx, left_values, left_prom):
        if not complete_cycle_check(left_signal, int(idx), valley=False, min_recovery=cfg["cycle_recovery"], window=cfg["cycle_window"]):
            continue
        left_keep.append(int(idx))
        left_quality.append(float(0.6 * np.clip((peak_value - cfg["height"]) / 0.8, 0.0, 1.0) + 0.4 * np.clip(prominence / 0.5, 0.0, 1.0)))

    right_keep: list[int] = []
    right_quality: list[float] = []
    for idx, peak_value, prominence in zip(right_idx, right_values, right_prom):
        if not complete_cycle_check(right_signal, int(idx), valley=False, min_recovery=cfg["cycle_recovery"], window=cfg["cycle_window"]):
            continue
        right_keep.append(int(idx))
        right_quality.append(float(0.6 * np.clip((peak_value - cfg["height"]) / 0.8, 0.0, 1.0) + 0.4 * np.clip(prominence / 0.5, 0.0, 1.0)))

    left_detected = len(left_keep) > 0
    right_detected = len(right_keep) > 0
    left_first = start_frame + left_keep[0] if left_detected else None
    right_first = start_frame + right_keep[0] if right_detected else None
    left_then_right = int(left_detected and right_detected and left_first is not None and right_first is not None and left_first < right_first)
    rep_count = int(left_detected) + int(right_detected)

    rep_status = "ok"
    if rep_count == 0:
        rep_status = "no_signal"
    elif left_then_right == 0:
        rep_status = "low_confidence"

    summary.update(
        {
            "rep_status": rep_status,
            "rep_count": rep_count,
            "rep_count_match": int(rep_count == 2 and left_then_right == 1),
            "event_count_left": len(left_keep),
            "event_count_right": len(right_keep),
            "quality_mean": safe_nanmean(left_quality + right_quality),
            "quality_min": safe_nanmin(left_quality + right_quality),
            "rep_left_raise_detected": int(left_detected),
            "rep_right_raise_detected": int(right_detected),
            "rep_left_then_right": left_then_right,
            "rep_peak_gap_frames": (right_first - left_first) if left_first is not None and right_first is not None else math.nan,
        }
    )

    event_frames: list[int] = []
    event_sides: list[str] = []
    event_quality: list[float] = []
    if left_first is not None:
        event_frames.append(left_first)
        event_sides.append("left")
        event_quality.append(left_quality[0] if left_quality else math.nan)
    if right_first is not None:
        event_frames.append(right_first)
        event_sides.append("right")
        event_quality.append(right_quality[0] if right_quality else math.nan)

    event_payload = {
        "action_id": 1,
        "signal_name": "left_wrist_above_head,right_wrist_above_head",
        "rep_count": rep_count,
        "expected_count": 2,
        "event_frames": event_frames,
        "event_sides": event_sides,
        "event_quality": event_quality,
        "extra_metrics": {
            "left_peak_count": len(left_keep),
            "right_peak_count": len(right_keep),
            "left_then_right": left_then_right,
            "peak_gap_frames": summary["rep_peak_gap_frames"],
        },
    }
    return summary, event_payload


def analyze_action2(video_id: str, action_name: str, segment_status: str, frame_window: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    summary = blank_summary(video_id, 2, action_name, segment_status, "theta_left,theta_right", "no_signal")
    if frame_window.empty:
        summary["rep_status"] = "no_bounds"
        return summary, {"action_id": 2, "signal_name": "theta_left,theta_right", "rep_count": None, "expected_count": 2, "event_frames": [], "event_sides": [], "event_quality": [], "extra_metrics": {}}

    theta_left = pd.to_numeric(frame_window["theta_left"], errors="coerce").to_numpy(dtype=np.float32)
    theta_right = pd.to_numeric(frame_window["theta_right"], errors="coerce").to_numpy(dtype=np.float32)
    arm_sync = pd.to_numeric(frame_window["arm_sync"], errors="coerce").to_numpy(dtype=np.float32)
    circle_range_left_best = get_numeric_series(frame_window, "circle_range_left_best", ("circle_range_left",))
    circle_range_right_best = get_numeric_series(frame_window, "circle_range_right_best", ("circle_range_right",))

    left_valid = theta_left[np.isfinite(theta_left)]
    right_valid = theta_right[np.isfinite(theta_right)]
    if left_valid.size < 2 and right_valid.size < 2:
        summary["rep_status"] = "no_signal"
        return summary, {"action_id": 2, "signal_name": "theta_left,theta_right", "rep_count": None, "expected_count": 2, "event_frames": [], "event_sides": [], "event_quality": [], "extra_metrics": {}}

    circle_count_left = float(np.abs(np.diff(left_valid)).sum() / (2.0 * math.pi)) if left_valid.size >= 2 else math.nan
    circle_count_right = float(np.abs(np.diff(right_valid)).sum() / (2.0 * math.pi)) if right_valid.size >= 2 else math.nan
    circle_count_mean = safe_nanmean([circle_count_left, circle_count_right])
    arm_sync_mean = safe_nanmean(list(arm_sync))
    range_left = safe_nanmean(list(circle_range_left_best))
    range_right = safe_nanmean(list(circle_range_right_best))

    quality_left = max(0.0, 1.0 - abs(circle_count_left - 2.0) / 2.0) if math.isfinite(circle_count_left) else math.nan
    quality_right = max(0.0, 1.0 - abs(circle_count_right - 2.0) / 2.0) if math.isfinite(circle_count_right) else math.nan
    sync_quality = max(0.0, min(1.0, (arm_sync_mean - 0.20) / 0.55)) if math.isfinite(arm_sync_mean) else math.nan
    combined_left_quality = quality_left * sync_quality if math.isfinite(quality_left) and math.isfinite(sync_quality) else math.nan
    combined_right_quality = quality_right * sync_quality if math.isfinite(quality_right) and math.isfinite(sync_quality) else math.nan

    rep_count = int(max(0, round(circle_count_mean))) if math.isfinite(circle_count_mean) else math.nan
    rep_count_match = int(math.isfinite(circle_count_mean) and abs(circle_count_mean - 2.0) <= 0.45 and math.isfinite(arm_sync_mean) and arm_sync_mean >= 0.30)
    rep_status = "ok"
    if not math.isfinite(circle_count_mean):
        rep_status = "no_signal"
    elif rep_count_match == 0:
        rep_status = "low_confidence"

    summary.update(
        {
            "rep_status": rep_status,
            "rep_count": rep_count,
            "rep_count_match": rep_count_match,
            "event_count_left": int(round(circle_count_left)) if math.isfinite(circle_count_left) else 0,
            "event_count_right": int(round(circle_count_right)) if math.isfinite(circle_count_right) else 0,
            "quality_mean": safe_nanmean([combined_left_quality, combined_right_quality]),
            "quality_min": safe_nanmin([combined_left_quality, combined_right_quality]),
            "rep_circle_count_left": circle_count_left,
            "rep_circle_count_right": circle_count_right,
            "rep_circle_count_mean": circle_count_mean,
            "rep_circle_sync_quality": arm_sync_mean,
            "rep_circle_range_left_best_mean": range_left,
            "rep_circle_range_right_best_mean": range_right,
        }
    )

    event_payload = {
        "action_id": 2,
        "signal_name": "theta_left,theta_right",
        "rep_count": rep_count if isinstance(rep_count, int) else None,
        "expected_count": 2,
        "event_frames": [],
        "event_sides": [],
        "event_quality": [combined_left_quality, combined_right_quality],
        "extra_metrics": {
            "circle_count_left": circle_count_left,
            "circle_count_right": circle_count_right,
            "circle_count_mean": circle_count_mean,
            "arm_sync_mean": arm_sync_mean,
            "circle_range_left_best_mean": range_left,
            "circle_range_right_best_mean": range_right,
        },
    }
    return summary, event_payload


def analyze_action3(video_id: str, action_name: str, segment_status: str, frame_window: pd.DataFrame, start_frame: int) -> tuple[dict[str, object], dict[str, object]]:
    summary = blank_summary(video_id, 3, action_name, segment_status, "mean_knee_angle", "no_signal")
    if frame_window.empty:
        summary["rep_status"] = "no_bounds"
        return summary, {"action_id": 3, "signal_name": "mean_knee_angle", "rep_count": None, "expected_count": 2, "event_frames": [], "event_sides": [], "event_quality": [], "extra_metrics": {}}

    cfg = ACTION_THRESHOLDS[3]
    knee_signal = pd.to_numeric(frame_window["mean_knee_angle"], errors="coerce").to_numpy(dtype=np.float32)
    hip_signal = pd.to_numeric(frame_window["center_y_delta"], errors="coerce").to_numpy(dtype=np.float32)

    valleys, valley_values, valley_prom = detect_extrema(knee_signal, valley=True, distance=cfg["min_distance"], prominence=cfg["prominence"])
    kept_frames: list[int] = []
    kept_quality: list[float] = []
    kept_knee_values: list[float] = []
    kept_hip_values: list[float] = []

    for idx, valley_value, prominence in zip(valleys, valley_values, valley_prom):
        hip_drop = float(hip_signal[int(idx)]) if np.isfinite(hip_signal[int(idx)]) else math.nan
        if not math.isfinite(hip_drop) or hip_drop < cfg["hip_drop"]:
            continue
        if not complete_cycle_check(knee_signal, int(idx), valley=True, min_recovery=cfg["cycle_recovery"], window=cfg["cycle_window"]):
            continue
        depth_score = np.clip((150.0 - valley_value) / 60.0, 0.0, 1.0)
        hip_score = np.clip((hip_drop - cfg["hip_drop"]) / 0.15, 0.0, 1.0)
        prominence_score = np.clip(prominence / 40.0, 0.0, 1.0)
        quality = float(0.45 * depth_score + 0.25 * hip_score + 0.30 * prominence_score)
        kept_frames.append(start_frame + int(idx))
        kept_quality.append(quality)
        kept_knee_values.append(float(valley_value))
        kept_hip_values.append(hip_drop)

    rep_count = len(kept_frames)
    rep_status = "ok"
    if rep_count == 0:
        rep_status = "no_signal"
    elif rep_count != 2 or safe_nanmean(kept_quality) < 0.35:
        rep_status = "low_confidence"

    stand_recovery = math.nan
    if len(kept_frames) >= 2:
        local_start = kept_frames[0] - start_frame
        local_end = kept_frames[1] - start_frame
        between = knee_signal[local_start:local_end + 1]
        valid_between = between[np.isfinite(between)]
        if valid_between.size:
            stand_recovery = float(valid_between.max() - min(kept_knee_values[0], kept_knee_values[1]))

    summary.update(
        {
            "rep_status": rep_status,
            "rep_count": rep_count,
            "rep_count_match": int(rep_count == 2),
            "event_count_left": 0,
            "event_count_right": 0,
            "quality_mean": safe_nanmean(kept_quality),
            "quality_min": safe_nanmin(kept_quality),
            "rep_squat_count": rep_count,
            "rep_rep1_frame": kept_frames[0] if len(kept_frames) >= 1 else math.nan,
            "rep_rep2_frame": kept_frames[1] if len(kept_frames) >= 2 else math.nan,
            "rep_min_knee_angle_rep1": kept_knee_values[0] if len(kept_knee_values) >= 1 else math.nan,
            "rep_min_knee_angle_rep2": kept_knee_values[1] if len(kept_knee_values) >= 2 else math.nan,
            "rep_center_y_delta_rep1": kept_hip_values[0] if len(kept_hip_values) >= 1 else math.nan,
            "rep_center_y_delta_rep2": kept_hip_values[1] if len(kept_hip_values) >= 2 else math.nan,
            "rep_stand_recovery_between_reps": stand_recovery,
        }
    )

    event_payload = {
        "action_id": 3,
        "signal_name": "mean_knee_angle",
        "rep_count": rep_count,
        "expected_count": 2,
        "event_frames": kept_frames,
        "event_sides": ["center"] * len(kept_frames),
        "event_quality": kept_quality,
        "extra_metrics": {
            "min_knee_angle_rep1": summary["rep_min_knee_angle_rep1"],
            "min_knee_angle_rep2": summary["rep_min_knee_angle_rep2"],
            "center_y_delta_rep1": summary["rep_center_y_delta_rep1"],
            "center_y_delta_rep2": summary["rep_center_y_delta_rep2"],
            "stand_recovery_between_reps": stand_recovery,
        },
    }
    return summary, event_payload


def detect_cross_touch_side(
    distance_signal: np.ndarray,
    bend_gate: np.ndarray,
    cfg: dict[str, float],
) -> tuple[list[int], list[float], list[float]]:
    valleys, values, prominences = detect_extrema(
        distance_signal,
        valley=True,
        distance=cfg["min_distance"],
        prominence=cfg["prominence"],
        valid_mask=(bend_gate >= cfg["bend_gate"]),
    )
    keep_idx: list[int] = []
    keep_quality: list[float] = []
    keep_values: list[float] = []
    for idx, value, prominence in zip(valleys, values, prominences):
        if not complete_cycle_check(distance_signal, int(idx), valley=True, min_recovery=cfg["cycle_recovery"], window=cfg["cycle_window"]):
            continue
        bend_score = float(bend_gate[int(idx)])
        distance_score = float(np.clip((1.30 - value) / 0.80, 0.0, 1.0))
        prominence_score = float(np.clip(prominence / 0.40, 0.0, 1.0))
        quality = float(0.45 * distance_score + 0.35 * bend_score + 0.20 * prominence_score)
        keep_idx.append(int(idx))
        keep_quality.append(quality)
        keep_values.append(float(value))
    return keep_idx, keep_quality, keep_values


def analyze_action4(video_id: str, action_name: str, segment_status: str, frame_window: pd.DataFrame, start_frame: int) -> tuple[dict[str, object], dict[str, object]]:
    summary = blank_summary(video_id, 4, action_name, segment_status, "dist_left_wrist_right_ankle,dist_right_wrist_left_ankle", "no_signal")
    if frame_window.empty:
        summary["rep_status"] = "no_bounds"
        return summary, {"action_id": 4, "signal_name": "dist_left_wrist_right_ankle,dist_right_wrist_left_ankle", "rep_count": None, "expected_count": 2, "event_frames": [], "event_sides": [], "event_quality": [], "extra_metrics": {}}

    cfg = ACTION_THRESHOLDS[4]
    left_signal = pd.to_numeric(frame_window["dist_left_wrist_right_ankle"], errors="coerce").to_numpy(dtype=np.float32)
    right_signal = pd.to_numeric(frame_window["dist_right_wrist_left_ankle"], errors="coerce").to_numpy(dtype=np.float32)
    shoulder_y = pd.to_numeric(frame_window["shoulder_center_y_norm"], errors="coerce").to_numpy(dtype=np.float32)
    nose_y = pd.to_numeric(frame_window["nose_y_norm"], errors="coerce").to_numpy(dtype=np.float32)
    shoulder_bend = clip_strength(shoulder_y, -0.70, -0.15)
    nose_bend = clip_strength(nose_y, -1.20, -0.15)
    bend_gate = safe_rowwise_nanmean(np.stack([shoulder_bend, nose_bend], axis=1))

    left_idx, left_quality, left_values = detect_cross_touch_side(left_signal, bend_gate, cfg)
    right_idx, right_quality, right_values = detect_cross_touch_side(right_signal, bend_gate, cfg)

    left_detected = len(left_idx) > 0
    right_detected = len(right_idx) > 0
    rep_count = int(left_detected) + int(right_detected)
    rep_status = "ok"
    if rep_count == 0:
        rep_status = "no_signal"
    elif rep_count < 2 or safe_nanmean(left_quality + right_quality) < 0.30:
        rep_status = "low_confidence"

    event_frames = [start_frame + idx for idx in left_idx] + [start_frame + idx for idx in right_idx]
    event_sides = (["left_to_right"] * len(left_idx)) + (["right_to_left"] * len(right_idx))
    event_quality = left_quality + right_quality

    summary.update(
        {
            "rep_status": rep_status,
            "rep_count": rep_count,
            "rep_count_match": int(rep_count == 2),
            "event_count_left": len(left_idx),
            "event_count_right": len(right_idx),
            "quality_mean": safe_nanmean(event_quality),
            "quality_min": safe_nanmin(event_quality),
            "rep_left_to_right_touch_detected": int(left_detected),
            "rep_right_to_left_touch_detected": int(right_detected),
            "rep_touch_count_total": len(left_idx) + len(right_idx),
            "rep_touch_symmetry": abs(len(left_idx) - len(right_idx)),
            "rep_min_cross_touch_left": min(left_values) if left_values else math.nan,
            "rep_min_cross_touch_right": min(right_values) if right_values else math.nan,
        }
    )

    event_payload = {
        "action_id": 4,
        "signal_name": "dist_left_wrist_right_ankle,dist_right_wrist_left_ankle",
        "rep_count": rep_count,
        "expected_count": 2,
        "event_frames": event_frames,
        "event_sides": event_sides,
        "event_quality": event_quality,
        "extra_metrics": {
            "left_to_right_touch_detected": int(left_detected),
            "right_to_left_touch_detected": int(right_detected),
            "min_cross_touch_left": summary["rep_min_cross_touch_left"],
            "min_cross_touch_right": summary["rep_min_cross_touch_right"],
            "touch_symmetry": summary["rep_touch_symmetry"],
        },
    }
    return summary, event_payload


def detect_side_bends(
    lateral_signal: np.ndarray,
    gate_signal: np.ndarray,
    cfg: dict[str, float],
) -> tuple[list[int], list[float], list[float]]:
    valid_mask = gate_signal >= cfg["gate"]
    peaks, values, prominences = detect_extrema(
        lateral_signal,
        valley=False,
        distance=cfg["min_distance"],
        prominence=cfg["prominence"],
        height=cfg["lateral"],
        valid_mask=valid_mask,
    )
    keep_idx: list[int] = []
    keep_quality: list[float] = []
    keep_values: list[float] = []
    for idx, value, prominence in zip(peaks, values, prominences):
        if not complete_cycle_check(lateral_signal, int(idx), valley=False, min_recovery=cfg["cycle_recovery"], window=cfg["cycle_window"]):
            continue
        gate_value = float(gate_signal[int(idx)])
        bend_score = float(np.clip((value - cfg["lateral"]) / 0.30, 0.0, 1.0))
        prominence_score = float(np.clip(prominence / 0.20, 0.0, 1.0))
        quality = float(0.50 * bend_score + 0.30 * gate_value + 0.20 * prominence_score)
        keep_idx.append(int(idx))
        keep_quality.append(quality)
        keep_values.append(float(value))
    return keep_idx, keep_quality, keep_values


def analyze_action5(video_id: str, action_name: str, segment_status: str, frame_window: pd.DataFrame, start_frame: int) -> tuple[dict[str, object], dict[str, object]]:
    summary = blank_summary(video_id, 5, action_name, segment_status, "shoulder_center_x_norm", "no_signal")
    if frame_window.empty:
        summary["rep_status"] = "no_bounds"
        return summary, {"action_id": 5, "signal_name": "shoulder_center_x_norm", "rep_count": None, "expected_count": 2, "event_frames": [], "event_sides": [], "event_quality": [], "extra_metrics": {}}

    cfg = ACTION_THRESHOLDS[5]
    shoulder_x = np.abs(pd.to_numeric(frame_window["shoulder_center_x_norm"], errors="coerce").to_numpy(dtype=np.float32))
    left_hip_dist = pd.to_numeric(frame_window["dist_left_wrist_left_hip"], errors="coerce").to_numpy(dtype=np.float32)
    right_hip_dist = pd.to_numeric(frame_window["dist_right_wrist_right_hip"], errors="coerce").to_numpy(dtype=np.float32)
    left_elbow_angle = pd.to_numeric(frame_window["left_elbow_angle"], errors="coerce").to_numpy(dtype=np.float32)
    right_elbow_angle = pd.to_numeric(frame_window["right_elbow_angle"], errors="coerce").to_numpy(dtype=np.float32)
    left_raise = pd.to_numeric(frame_window["left_wrist_above_head"], errors="coerce").to_numpy(dtype=np.float32)
    right_raise = pd.to_numeric(frame_window["right_wrist_above_head"], errors="coerce").to_numpy(dtype=np.float32)

    left_hand_on_hip = descending_strength(left_hip_dist, 0.35, 1.15)
    right_hand_on_hip = descending_strength(right_hip_dist, 0.35, 1.15)
    left_elbow_bent = descending_strength(left_elbow_angle, 120.0, 165.0)
    right_elbow_bent = descending_strength(right_elbow_angle, 120.0, 165.0)
    left_raise_strength = clip_strength(left_raise, 0.08, 0.90)
    right_raise_strength = clip_strength(right_raise, 0.08, 0.90)

    left_gate = np.minimum(left_hand_on_hip * left_elbow_bent, right_raise_strength).astype(np.float32)
    right_gate = np.minimum(right_hand_on_hip * right_elbow_bent, left_raise_strength).astype(np.float32)

    left_idx, left_quality, left_values = detect_side_bends(shoulder_x, left_gate, cfg)
    right_idx, right_quality, right_values = detect_side_bends(shoulder_x, right_gate, cfg)

    left_detected = len(left_idx) > 0
    right_detected = len(right_idx) > 0
    rep_count = int(left_detected) + int(right_detected)
    rep_status = "ok"
    if rep_count == 0:
        rep_status = "no_signal"
    elif rep_count < 2 or safe_nanmean(left_quality + right_quality) < 0.30:
        rep_status = "low_confidence"

    event_frames = [start_frame + idx for idx in left_idx] + [start_frame + idx for idx in right_idx]
    event_sides = (["left"] * len(left_idx)) + (["right"] * len(right_idx))
    event_quality = left_quality + right_quality

    summary.update(
        {
            "rep_status": rep_status,
            "rep_count": rep_count,
            "rep_count_match": int(rep_count == 2),
            "event_count_left": len(left_idx),
            "event_count_right": len(right_idx),
            "quality_mean": safe_nanmean(event_quality),
            "quality_min": safe_nanmin(event_quality),
            "rep_side_bend_left_detected": int(left_detected),
            "rep_side_bend_right_detected": int(right_detected),
            "rep_side_bend_count": len(left_idx) + len(right_idx),
            "rep_max_bend_left": max(left_values) if left_values else math.nan,
            "rep_max_bend_right": max(right_values) if right_values else math.nan,
        }
    )

    event_payload = {
        "action_id": 5,
        "signal_name": "shoulder_center_x_norm",
        "rep_count": rep_count,
        "expected_count": 2,
        "event_frames": event_frames,
        "event_sides": event_sides,
        "event_quality": event_quality,
        "extra_metrics": {
            "side_bend_left_detected": int(left_detected),
            "side_bend_right_detected": int(right_detected),
            "max_bend_left": summary["rep_max_bend_left"],
            "max_bend_right": summary["rep_max_bend_right"],
        },
    }
    return summary, event_payload


def analyze_segment(video_id: str, segment_row: pd.Series, frame_window: pd.DataFrame, start_frame: int | None) -> tuple[dict[str, object], dict[str, object]]:
    action_id = int(segment_row["action_id"])
    action_name = str(segment_row.get("action_name", ACTION_NAMES.get(action_id, f"action_{action_id}")))
    segment_status = str(segment_row.get("status", ""))
    if start_frame is None:
        summary = blank_summary(video_id, action_id, action_name, segment_status, "", "no_bounds")
        summary["rep_status"] = "no_bounds"
        event_payload = {
            "action_id": action_id,
            "signal_name": "",
            "rep_count": None,
            "expected_count": EXPECTED_REP_COUNT[action_id],
            "event_frames": [],
            "event_sides": [],
            "event_quality": [],
            "extra_metrics": {},
        }
        return summary, event_payload

    if action_id == 1:
        return analyze_action1(video_id, action_name, segment_status, frame_window, start_frame)
    if action_id == 2:
        return analyze_action2(video_id, action_name, segment_status, frame_window)
    if action_id == 3:
        return analyze_action3(video_id, action_name, segment_status, frame_window, start_frame)
    if action_id == 4:
        return analyze_action4(video_id, action_name, segment_status, frame_window, start_frame)
    if action_id == 5:
        return analyze_action5(video_id, action_name, segment_status, frame_window, start_frame)
    raise ValueError(f"Unsupported action_id: {action_id}")


def process_video_id(video_id: str, output_dir: Path) -> tuple[Path, Path]:
    frame_features_dir, segments_dir, rep_events_dir, rep_summary_dir = ensure_rep_dirs(output_dir)
    frame_features_path = build_frame_features_path(video_id, frame_features_dir)
    segments_path = build_segments_path(video_id, segments_dir)
    rep_events_path = build_rep_events_path(video_id, rep_events_dir)
    rep_summary_path = build_rep_summary_path(video_id, rep_summary_dir)

    if not frame_features_path.exists():
        raise FileNotFoundError(f"Frame features not found: {frame_features_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments not found: {segments_path}")

    frame_features_df = pd.read_csv(frame_features_path)
    segments_df = pd.read_csv(segments_path)

    summary_rows: list[dict[str, object]] = []
    action_events: list[dict[str, object]] = []
    for _, segment_row in segments_df.iterrows():
        frame_window, start_frame, _ = get_segment_window(frame_features_df, segment_row)
        summary_row, event_payload = analyze_segment(video_id, segment_row, frame_window, start_frame)
        summary_rows.append(summary_row)
        action_events.append(event_payload)

    rep_summary_df = pd.DataFrame(summary_rows).sort_values(["action_id"]).reset_index(drop=True)
    rep_summary_df.to_csv(rep_summary_path, index=False)
    write_json(
        rep_events_path,
        {
            "video_id": video_id,
            "action_events": action_events,
        },
    )
    return rep_events_path, rep_summary_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    video_ids = collect_video_ids(args)
    if not video_ids:
        raise SystemExit("Khong tim thay video/segment nao de tinh rep count.")

    print(f"Se tinh rep count cho {len(video_ids)} video.")
    for index, video_id in enumerate(video_ids, start=1):
        if not args.force and rep_outputs_ready(video_id, output_dir):
            print(f"[{index}/{len(video_ids)}] Skip {video_id} (rep artifacts already exist)")
            continue
        print(f"[{index}/{len(video_ids)}] Processing {video_id} ...")
        rep_events_path, rep_summary_path = process_video_id(video_id, output_dir)
        print(f"  Rep events: {rep_events_path}")
        print(f"  Rep summary: {rep_summary_path}")


if __name__ == "__main__":
    main()
