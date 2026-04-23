from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from action_training_common import (
    BONE_CONNECTIONS,
    DEFAULT_DATASET_DIRNAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEQUENCE_LENGTH,
    SELECTED_LANDMARK_INDICES,
    SELECTED_LANDMARK_NAMES,
    SUMMARY_SOURCE_COLUMNS,
    collect_segment_csv_paths,
    compute_bone_vectors,
    compute_motion,
    count_signal_peaks,
    ensure_dir,
    nan_summary_stats,
    parse_boolish,
    random_video_split,
    resample_temporal_array,
    write_json,
)
from segment_pose_routine import parse_target_bits, load_normalized_pose_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an action-level dataset manifest and sequence tensors from normalized pose, frame features, and segments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory containing normalized_skeleton, frame_features, and segments.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory for action dataset artifacts. Default: <output-dir>/action_dataset",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Target number of frames per action sequence. Default: {DEFAULT_SEQUENCE_LENGTH}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for video-level split assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio. Default: 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio. Default: 0.15",
    )
    return parser.parse_args()


def compute_visibility_quality(visibility: np.ndarray) -> dict[str, float]:
    arr = np.asarray(visibility, dtype=np.float32).reshape(-1)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {
            "feat_visibility_quality_mean": math.nan,
            "feat_visibility_quality_std": math.nan,
            "feat_visibility_quality_min": math.nan,
        }
    return {
        "feat_visibility_quality_mean": float(valid.mean()),
        "feat_visibility_quality_std": float(valid.std()),
        "feat_visibility_quality_min": float(valid.min()),
    }


def mean_abs_pair_gap(frame_window: pd.DataFrame, left_col: str, right_col: str) -> float:
    if left_col not in frame_window.columns or right_col not in frame_window.columns:
        return math.nan
    left = pd.to_numeric(frame_window[left_col], errors="coerce").to_numpy(dtype=np.float32)
    right = pd.to_numeric(frame_window[right_col], errors="coerce").to_numpy(dtype=np.float32)
    diff = np.abs(left - right)
    valid = np.isfinite(diff)
    if not np.any(valid):
        return math.nan
    return float(diff[valid].mean())


def build_feature_summary(
    frame_window: pd.DataFrame,
    visibility_segment: np.ndarray,
    action_id: int,
    duration_frames: int,
    duration_ms: int | None,
) -> dict[str, float]:
    summary: dict[str, float] = {
        "feat_duration_frames": float(duration_frames),
        "feat_duration_ms": math.nan if duration_ms is None else float(duration_ms),
    }

    if frame_window.empty:
        for source_col in SUMMARY_SOURCE_COLUMNS:
            for stat_name in ("mean", "std", "min", "max", "range"):
                summary[f"feat_{source_col}_{stat_name}"] = math.nan
        summary["feat_present_ratio"] = math.nan
        summary["feat_symmetry_elbow_mean_abs_diff"] = math.nan
        summary["feat_symmetry_knee_mean_abs_diff"] = math.nan
        summary["feat_symmetry_hand_height_mean_abs_diff"] = math.nan
        summary["feat_action_signal_peak_count"] = math.nan
        summary["feat_action_signal_active_ratio"] = math.nan
        summary["feat_action_signal_mean"] = math.nan
        summary["feat_action_signal_max"] = math.nan
        summary.update(compute_visibility_quality(visibility_segment))
        return summary

    reindexed = frame_window.reindex(columns=SUMMARY_SOURCE_COLUMNS)
    for source_col in SUMMARY_SOURCE_COLUMNS:
        values = pd.to_numeric(reindexed[source_col], errors="coerce").to_numpy(dtype=np.float32)
        for stat_name, stat_value in nan_summary_stats(values).items():
            summary[f"feat_{source_col}_{stat_name}"] = stat_value

    person_present = pd.to_numeric(frame_window.get("person_present", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float32)
    if person_present.size == 0:
        summary["feat_present_ratio"] = math.nan
    else:
        valid_present = np.isfinite(person_present)
        summary["feat_present_ratio"] = float(person_present[valid_present].mean()) if np.any(valid_present) else math.nan

    summary["feat_symmetry_elbow_mean_abs_diff"] = mean_abs_pair_gap(frame_window, "left_elbow_angle", "right_elbow_angle")
    summary["feat_symmetry_knee_mean_abs_diff"] = mean_abs_pair_gap(frame_window, "left_knee_angle", "right_knee_angle")
    summary["feat_symmetry_hand_height_mean_abs_diff"] = mean_abs_pair_gap(frame_window, "left_wrist_above_head", "right_wrist_above_head")

    action_score_col = f"action_{action_id}_score"
    if action_score_col in frame_window.columns:
        action_score = pd.to_numeric(frame_window[action_score_col], errors="coerce").to_numpy(dtype=np.float32)
        valid_score = action_score[np.isfinite(action_score)]
        summary["feat_action_signal_peak_count"] = float(count_signal_peaks(valid_score, threshold=0.5)) if valid_score.size else math.nan
        summary["feat_action_signal_active_ratio"] = float(np.mean(valid_score >= 0.5)) if valid_score.size else math.nan
        summary["feat_action_signal_mean"] = float(valid_score.mean()) if valid_score.size else math.nan
        summary["feat_action_signal_max"] = float(valid_score.max()) if valid_score.size else math.nan
    else:
        summary["feat_action_signal_peak_count"] = math.nan
        summary["feat_action_signal_active_ratio"] = math.nan
        summary["feat_action_signal_mean"] = math.nan
        summary["feat_action_signal_max"] = math.nan

    candidate_col = f"action_{action_id}_candidate"
    if candidate_col in frame_window.columns:
        candidate = pd.to_numeric(frame_window[candidate_col], errors="coerce").to_numpy(dtype=np.float32)
        valid_candidate = candidate[np.isfinite(candidate)]
        summary["feat_action_candidate_ratio"] = float(valid_candidate.mean()) if valid_candidate.size else math.nan
    else:
        summary["feat_action_candidate_ratio"] = math.nan

    summary.update(compute_visibility_quality(visibility_segment))
    return summary


def build_sequence_streams(
    coords_segment: np.ndarray,
    visibility_segment: np.ndarray,
    target_length: int,
) -> dict[str, np.ndarray]:
    selected_joints = coords_segment[:, SELECTED_LANDMARK_INDICES, :]
    selected_visibility = visibility_segment[:, SELECTED_LANDMARK_INDICES]

    joints_resampled = resample_temporal_array(selected_joints, target_length)
    visibility_resampled = resample_temporal_array(selected_visibility, target_length)

    bones_resampled = compute_bone_vectors(joints_resampled)
    motion_resampled = compute_motion(joints_resampled)
    bone_motion_resampled = compute_motion(bones_resampled)

    return {
        "joint": np.nan_to_num(joints_resampled, nan=0.0).astype(np.float32),
        "visibility": np.nan_to_num(visibility_resampled, nan=0.0).astype(np.float32),
        "bone": np.nan_to_num(bones_resampled, nan=0.0).astype(np.float32),
        "motion": np.nan_to_num(motion_resampled, nan=0.0).astype(np.float32),
        "bone_motion": np.nan_to_num(bone_motion_resampled, nan=0.0).astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    dataset_dir = (args.dataset_dir.resolve() if args.dataset_dir else (output_dir / DEFAULT_DATASET_DIRNAME).resolve())
    normalized_dir = output_dir / "normalized_skeleton"
    frame_features_dir = output_dir / "frame_features"
    rep_summary_dir = output_dir / "rep_summary"

    sequence_dir = ensure_dir(dataset_dir / "sequence_tensors")
    manifest_path = dataset_dir / "action_dataset_manifest.csv"
    summary_path = dataset_dir / "action_dataset_summary.json"

    rows: list[dict[str, object]] = []
    labeled_video_ids: list[str] = []

    segment_paths = collect_segment_csv_paths(output_dir)
    if not segment_paths:
        raise SystemExit(f"Khong tim thay file segment nao trong {output_dir}")

    print(f"Se build action dataset tu {len(segment_paths)} segment files.")

    for segment_path in segment_paths:
        video_id = segment_path.stem.replace("_segments", "")
        normalized_path = normalized_dir / f"{video_id}_pose_keypoints_normalized.csv"
        frame_features_path = frame_features_dir / f"{video_id}_frame_features.csv"
        rep_summary_path = rep_summary_dir / f"{video_id}_rep_summary.csv"
        video_target_bits = parse_target_bits(Path(video_id))

        segments_df = pd.read_csv(segment_path)
        sequence = load_normalized_pose_csv(normalized_path) if normalized_path.exists() else None
        frame_features_df = pd.read_csv(frame_features_path) if frame_features_path.exists() else None
        rep_summary_by_action: dict[int, dict[str, object]] = {}
        if rep_summary_path.exists():
            rep_summary_df = pd.read_csv(rep_summary_path)
            for _, rep_row in rep_summary_df.iterrows():
                rep_action_id = int(rep_row["action_id"])
                rep_summary_by_action[rep_action_id] = {
                    (column_name if column_name.startswith("rep_") else f"rep_{column_name}"): rep_row[column_name]
                    for column_name in rep_summary_df.columns
                    if column_name not in {"video_id", "action_id"}
                }

        if sequence is not None:
            max_frame_index = max(sequence.coords_norm.shape[0] - 1, 0)
        elif frame_features_df is not None and not frame_features_df.empty:
            max_frame_index = int(pd.to_numeric(frame_features_df["frame_idx"], errors="coerce").max())
        else:
            max_frame_index = -1

        for _, row in segments_df.iterrows():
            action_id = int(row["action_id"])
            action_name = str(row["action_name"])
            target_label_value = row.get("target_label", "")
            if pd.isna(target_label_value) or str(target_label_value).strip() == "":
                target_label = video_target_bits[action_id - 1] if len(video_target_bits) == 5 else ""
            else:
                target_label = str(target_label_value).strip()

            is_labeled = target_label in {"0", "1"}
            if is_labeled:
                labeled_video_ids.append(video_id)

            has_bounds = pd.notna(row.get("start_frame")) and pd.notna(row.get("end_frame"))
            start_frame = int(row["start_frame"]) if has_bounds else None
            end_frame = int(row["end_frame"]) if has_bounds else None
            build_status = "ok"
            window_clipped = False
            used_start_frame = start_frame
            used_end_frame = end_frame

            if not normalized_path.exists():
                build_status = "missing_normalized_pose"
            elif not frame_features_path.exists():
                build_status = "missing_frame_features"
            elif not has_bounds:
                build_status = "no_bounds"
            else:
                assert start_frame is not None and end_frame is not None
                used_start_frame = max(0, start_frame)
                used_end_frame = min(end_frame, max_frame_index)
                window_clipped = (used_start_frame != start_frame) or (used_end_frame != end_frame)
                if used_end_frame < used_start_frame:
                    build_status = "empty_window"

            duration_frames = 0
            duration_ms: int | None = None
            sequence_tensor_relpath = ""
            feature_summary: dict[str, float] = {}

            if build_status == "ok":
                assert sequence is not None and frame_features_df is not None
                assert used_start_frame is not None and used_end_frame is not None
                frame_mask = (frame_features_df["frame_idx"] >= used_start_frame) & (frame_features_df["frame_idx"] <= used_end_frame)
                frame_window = frame_features_df.loc[frame_mask].copy()

                coords_segment = sequence.coords_norm[used_start_frame : used_end_frame + 1]
                visibility_segment = sequence.visibility[used_start_frame : used_end_frame + 1]
                timestamps_segment = sequence.timestamps_ms[used_start_frame : used_end_frame + 1]

                duration_frames = int(coords_segment.shape[0])
                if duration_frames <= 0 or frame_window.empty:
                    build_status = "empty_window"
                else:
                    duration_ms = int(timestamps_segment[-1] - timestamps_segment[0]) if timestamps_segment.size >= 2 else 0
                    feature_summary = build_feature_summary(
                        frame_window=frame_window,
                        visibility_segment=visibility_segment[:, SELECTED_LANDMARK_INDICES],
                        action_id=action_id,
                        duration_frames=duration_frames,
                        duration_ms=duration_ms,
                    )
                    sample_id = f"{video_id}__action_{action_id}"
                    sequence_tensor_relpath = f"sequence_tensors/{sample_id}.npz"
                    sequence_tensor_path = dataset_dir / sequence_tensor_relpath
                    streams = build_sequence_streams(coords_segment, visibility_segment, args.sequence_length)
                    np.savez_compressed(
                        sequence_tensor_path,
                        joint=streams["joint"],
                        visibility=streams["visibility"],
                        bone=streams["bone"],
                        motion=streams["motion"],
                        bone_motion=streams["bone_motion"],
                        action_id=np.asarray([action_id], dtype=np.int64),
                        target_label=np.asarray([int(target_label)]) if is_labeled else np.asarray([], dtype=np.int64),
                    )
            else:
                feature_summary = build_feature_summary(
                    frame_window=pd.DataFrame(columns=SUMMARY_SOURCE_COLUMNS),
                    visibility_segment=np.empty((0, len(SELECTED_LANDMARK_INDICES)), dtype=np.float32),
                    action_id=action_id,
                    duration_frames=0,
                    duration_ms=None,
                )

            manifest_row: dict[str, object] = {
                "sample_id": f"{video_id}__action_{action_id}",
                "video_id": video_id,
                "video_target_bits": video_target_bits,
                "action_id": action_id,
                "action_name": action_name,
                "target_label": target_label,
                "is_labeled": int(is_labeled),
                "segment_status": str(row.get("status", "")),
                "segment_source": str(row.get("segment_source", segment_path.parent.name)),
                "missing": int(parse_boolish(row.get("missing", False))),
                "start_frame": used_start_frame if used_start_frame is not None else "",
                "end_frame": used_end_frame if used_end_frame is not None else "",
                "start_ms": "" if pd.isna(row.get("start_ms")) else int(row.get("start_ms")),
                "end_ms": "" if pd.isna(row.get("end_ms")) else int(row.get("end_ms")),
                "num_frames": duration_frames,
                "window_clipped": int(window_clipped),
                "has_bounds": int(has_bounds),
                "sequence_ready": int(build_status == "ok"),
                "build_status": build_status,
                "split": "unassigned",
                "sequence_tensor_path": sequence_tensor_relpath,
                "sequence_length": args.sequence_length,
                "selected_joint_count": len(SELECTED_LANDMARK_INDICES),
                "selected_bone_count": len(BONE_CONNECTIONS),
            }
            manifest_row.update(feature_summary)
            manifest_row.update(rep_summary_by_action.get(action_id, {}))
            rows.append(manifest_row)

    split_map = random_video_split(
        video_ids=sorted(set(labeled_video_ids)),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    for row in rows:
        if not row["is_labeled"]:
            row["split"] = "unlabeled"
        else:
            row["split"] = split_map.get(str(row["video_id"]), "train")

    manifest_df = pd.DataFrame(rows)
    manifest_df = manifest_df.sort_values(["video_id", "action_id"]).reset_index(drop=True)
    ensure_dir(dataset_dir)
    manifest_df.to_csv(manifest_path, index=False)

    usable_df = manifest_df[(manifest_df["is_labeled"] == 1) & (manifest_df["sequence_ready"] == 1)]
    summary_payload = {
        "manifest_path": str(manifest_path),
        "sequence_dir": str(sequence_dir),
        "sequence_length": args.sequence_length,
        "selected_joint_names": SELECTED_LANDMARK_NAMES,
        "selected_bone_connections": list(BONE_CONNECTIONS),
        "total_rows": int(len(manifest_df)),
        "usable_rows": int(len(usable_df)),
        "labeled_video_count": int(len(split_map)),
        "split_counts": {
            split_name: int(count)
            for split_name, count in manifest_df["split"].value_counts(dropna=False).sort_index().items()
        },
        "usable_split_counts": {
            split_name: int(count)
            for split_name, count in usable_df["split"].value_counts(dropna=False).sort_index().items()
        },
        "action_target_distribution": {
            f"action_{action_id}": {
                str(label): int(count)
                for label, count in subset["target_label"].value_counts(dropna=False).sort_index().items()
            }
            for action_id, subset in usable_df.groupby("action_id", sort=True)
        },
        "build_status_counts": {
            str(status): int(count)
            for status, count in manifest_df["build_status"].value_counts(dropna=False).sort_index().items()
        },
        "rep_summary_rows_merged": int(manifest_df["rep_status"].notna().sum()) if "rep_status" in manifest_df.columns else 0,
    }
    write_json(summary_path, summary_payload)

    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Usable labeled rows: {len(usable_df)} / {len(manifest_df)}")


if __name__ == "__main__":
    main()
