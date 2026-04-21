from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from test_pose_extract import (
    VISIBILITY_THRESHOLD,
    draw_pose_from_pixel_points,
    interpolate_series,
    list_available_videos,
    resolve_input_paths,
    smooth_series,
    split_present_segments,
    start_ffmpeg_writer,
)

# onset = frame where action first appears and sustains for at least MIN_RUN_FRAMES
DEFAULT_VIDEO_DIR = Path("video_data")
DEFAULT_OUTPUT_DIR = Path("output")
NUM_LANDMARKS = 33 
MIN_RUN_FRAMES = 5 # mot action phai xuat hien lien tuc it nhat 5 frame
SKIP_LOOKAHEAD_FRAMES = 18 
SEARCH_GAP_FRAMES = 12 # sau khi phat hien mot action, bo qua it nhat 12 frame truoc khi tim action tiep theo, de tranh trung lap va phan bo on dinh hon
FEATURE_SMOOTH_WINDOW = 5
ROLLING_WINDOW = 15
ACTION_2_RESCUE_MIN_RUN = 3
ACTION_2_RESCUE_MIN_RANGE = 1.8
ACTION_2_RESCUE_MIN_SYNC = 0.75
ACTION_2_RESCUE_MIN_SPEED = 0.12
ACTION_2_RESCUE_MAX_ACTION1 = 0.35
ACTION_2_RESCUE_MIN_KNEE_ANGLE = 155.0

NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_ELBOW_IDX = 13
RIGHT_ELBOW_IDX = 14
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_KNEE_IDX = 25
RIGHT_KNEE_IDX = 26
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28

ACTION_NAMES = {
    1: "raise_lr",
    2: "arm_circle",
    3: "squat",
    4: "cross_touch",
    5: "side_bend",
}

ACTION_DISPLAY_NAMES = {
    1: "A1 Raise L/R",
    2: "A2 Arm Circle",
    3: "A3 Squat",
    4: "A4 Cross Touch",
    5: "A5 Side Bend",
}

ACTION_COLORS = {
    1: (74, 139, 255),
    2: (63, 204, 178),
    3: (69, 181, 101),
    4: (0, 170, 255),
    5: (87, 96, 255),
}
# todo: tuning
ACTION_THRESHOLDS = {
    1: 0.55,
    2: 0.48,
    3: 0.45,
    4: 0.48,
    5: 0.55,
}

LABEL_PATTERN = re.compile(r"(?:^|_)([01]{5})$")


@dataclass
class PoseSequence: # dataclass nay chua toan bo chuoi pose da norm
    timestamps_ms: np.ndarray
    person_present: np.ndarray
    coords_norm: np.ndarray
    visibility: np.ndarray
    centers: np.ndarray
    scales: np.ndarray


@dataclass
class SegmentRecord: # 1 action slot sau khi cat, chua thong tin ve action do nhu id, ten, trang thai, confidence, quy tac onset offset, v.v.
    action_id: int
    action_name: str
    status: str
    missing: bool
    start_frame: int | None
    end_frame: int | None
    start_ms: int | None
    end_ms: int | None
    confidence: float
    target_label: str # ground truth dong tac 0 or 1
    onset_rule: str # luật (rule) dùng để xác định thời điểm bắt đầu (onset)
    offset_rule: str # luật xác định thời điểm kết thúc (offset) của động tác


@dataclass
class DetectionHit:
    onset_frame: int
    confidence: float
    onset_rule: str
    status: str = "detected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment normalized pose sequences into 5 ordered action slots and render QA overlays."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help=(
            "Video files or directories. Relative paths are resolved under --video-dir. "
            "If empty, the script prompts from keyboard."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory used to resolve typed filenames. Default: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root output directory. The script reads normalized poses from "
            "<output-dir>/normalized_skeleton and writes artifacts to "
            "<output-dir>/frame_features, <output-dir>/segments, and <output-dir>/segmentation_qa."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute segmentation artifacts even if they already exist.",
    )
    return parser.parse_args()


def prompt_for_videos(video_dir: Path) -> list[Path]:
    available_videos = list_available_videos(video_dir)
    if available_videos:
        print("Available videos:")
        for path in available_videos:
            print(f"  - {path}")

    print("Nhap file video hoac thu muc video tren moi dong, hoac nhap nhieu muc cach nhau bang dau phay.")
    print("Go 'all' de chay tat ca video trong video_data. Enter trong de bat dau.")

    selected: list[Path] = []
    seen: set[Path] = set()

    while True:
        raw = input("video> ").strip()
        if not raw:
            break
        if raw.lower() == "all":
            return available_videos

        for token in [part.strip() for part in raw.split(",") if part.strip()]:
            try:
                resolved_paths = resolve_input_paths(token, video_dir)
            except FileNotFoundError as exc:
                print(exc)
                continue

            for resolved in resolved_paths:
                if resolved not in seen:
                    selected.append(resolved)
                    seen.add(resolved)

    return selected


def collect_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.videos:
        selected: list[Path] = []
        seen: set[Path] = set()
        for input_arg in args.videos:
            for resolved in resolve_input_paths(input_arg, args.video_dir):
                if resolved not in seen:
                    selected.append(resolved)
                    seen.add(resolved)
        return selected

    selected = prompt_for_videos(args.video_dir)
    if not selected:
        raise SystemExit("Khong co video nao duoc chon.")
    return selected

# tao cac thu muc can thiet de luu ket qua segment va QA, tra ve cac path tuong ung
def ensure_segmentation_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    normalized_dir = output_dir / "normalized_skeleton"
    frame_features_dir = output_dir / "frame_features"
    segments_dir = output_dir / "segments"
    qa_dir = output_dir / "segmentation_qa"
    frame_features_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    return normalized_dir, frame_features_dir, segments_dir, qa_dir


def build_normalized_input_path(video_path: Path, normalized_dir: Path) -> Path:
    return normalized_dir / f"{video_path.stem}_pose_keypoints_normalized.csv"


def build_frame_features_path(video_path: Path, frame_features_dir: Path) -> Path:
    return frame_features_dir / f"{video_path.stem}_frame_features.csv"


def build_segments_path(video_path: Path, segments_dir: Path) -> Path:
    return segments_dir / f"{video_path.stem}_segments.csv"


def build_segmentation_qa_path(video_path: Path, qa_dir: Path) -> Path:
    return qa_dir / f"{video_path.stem}_segmentation_overlay.mp4"

# build duong dan den cac file dau vao va dau ra can thiet cho viec segment pose, tra ve 4 path tuong ung
def resolve_output_paths(video_path: Path, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    normalized_dir, frame_features_dir, segments_dir, qa_dir = ensure_segmentation_dirs(output_dir)
    normalized_path = build_normalized_input_path(video_path, normalized_dir)
    frame_features_path = build_frame_features_path(video_path, frame_features_dir)
    segments_path = build_segments_path(video_path, segments_dir)
    qa_path = build_segmentation_qa_path(video_path, qa_dir)
    return normalized_path, frame_features_path, segments_path, qa_path


def outputs_ready(output_paths: tuple[Path, Path, Path, Path]) -> bool:
    _, frame_features_path, segments_path, qa_path = output_paths
    return frame_features_path.exists() and segments_path.exists() and qa_path.exists()


def parse_target_bits(video_path: Path) -> str:
    match = LABEL_PATTERN.search(video_path.stem)
    return match.group(1) if match else ""


def empty_coord_frame() -> np.ndarray:
    return np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)


def empty_visibility_frame() -> np.ndarray:
    return np.full(NUM_LANDMARKS, np.nan, dtype=np.float32)


def load_normalized_pose_csv(csv_path: Path) -> PoseSequence:
    rows_by_frame: dict[int, dict[str, object]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            timestamp_ms = int(row["timestamp_ms"])
            person_present = int(row["person_present"])

            frame = rows_by_frame.setdefault(
                frame_idx,
                {
                    "timestamp_ms": timestamp_ms,
                    "person_present": 0,
                    "coords": empty_coord_frame(),
                    "visibility": empty_visibility_frame(),
                    "center": np.full(3, np.nan, dtype=np.float32),
                    "scale": np.nan,
                },
            )

            frame["timestamp_ms"] = timestamp_ms
            if person_present != 1:
                continue

            frame["person_present"] = 1
            center = frame["center"]
            center[0] = np.nan if not row["center_x"] else float(row["center_x"])
            center[1] = np.nan if not row["center_y"] else float(row["center_y"])
            center[2] = np.nan if not row["center_z"] else float(row["center_z"])
            frame["scale"] = np.nan if not row["scale"] else float(row["scale"])

            landmark_idx = int(row["landmark_idx"])
            if landmark_idx < 0:
                continue

            coords = frame["coords"]
            visibility = frame["visibility"]
            coords[landmark_idx, 0] = np.nan if not row["x"] else float(row["x"])
            coords[landmark_idx, 1] = np.nan if not row["y"] else float(row["y"])
            coords[landmark_idx, 2] = np.nan if not row["z"] else float(row["z"])
            visibility[landmark_idx] = np.nan if not row["visibility"] else float(row["visibility"])

    if not rows_by_frame:
        return PoseSequence(
            timestamps_ms=np.empty(0, dtype=np.int32),
            person_present=np.empty(0, dtype=np.int8),
            coords_norm=np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32),
            visibility=np.empty((0, NUM_LANDMARKS), dtype=np.float32),
            centers=np.empty((0, 3), dtype=np.float32),
            scales=np.empty(0, dtype=np.float32),
        )

    num_frames = max(rows_by_frame) + 1
    timestamps_ms = np.zeros(num_frames, dtype=np.int32)
    person_present = np.zeros(num_frames, dtype=np.int8)
    coords_norm = np.full((num_frames, NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    visibility = np.full((num_frames, NUM_LANDMARKS), np.nan, dtype=np.float32)
    centers = np.full((num_frames, 3), np.nan, dtype=np.float32)
    scales = np.full(num_frames, np.nan, dtype=np.float32)

    for frame_idx, frame in rows_by_frame.items():
        timestamps_ms[frame_idx] = int(frame["timestamp_ms"])
        person_present[frame_idx] = int(frame["person_present"])
        coords_norm[frame_idx] = frame["coords"]
        visibility[frame_idx] = frame["visibility"]
        centers[frame_idx] = frame["center"]
        scales[frame_idx] = frame["scale"]

    return PoseSequence(
        timestamps_ms=timestamps_ms,
        person_present=person_present,
        coords_norm=coords_norm,
        visibility=visibility,
        centers=centers,
        scales=scales,
    )

# tinh toan diem trung binh cua cac landmark duoc chon tren moi frame, tra ve mang 2D voi shape (num_frames, num_dims)
def mean_points(points: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(points), axis=2)
    counts = valid.sum(axis=1)
    summed = np.nansum(points, axis=1)
    out = np.full((points.shape[0], points.shape[2]), np.nan, dtype=np.float32)
    valid_rows = counts > 0
    if np.any(valid_rows):
        out[valid_rows] = summed[valid_rows] / counts[valid_rows, None]
    return out


def rowwise_nanmean(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values, axis=1).astype(np.float32)
    valid = np.isfinite(stacked)
    counts = valid.sum(axis=1)
    summed = np.nansum(stacked, axis=1)
    out = np.full(stacked.shape[0], np.nan, dtype=np.float32)
    valid_rows = counts > 0
    if np.any(valid_rows):
        out[valid_rows] = summed[valid_rows] / counts[valid_rows]
    return out


def rowwise_nanmin(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values, axis=1).astype(np.float32)
    valid = np.isfinite(stacked)
    out = np.full(stacked.shape[0], np.nan, dtype=np.float32)
    valid_rows = np.any(valid, axis=1)
    if np.any(valid_rows):
        filtered = np.where(valid[valid_rows], stacked[valid_rows], np.inf)
        out[valid_rows] = filtered.min(axis=1)
    return out


def rowwise_nanmax(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values, axis=1).astype(np.float32)
    valid = np.isfinite(stacked)
    out = np.full(stacked.shape[0], np.nan, dtype=np.float32)
    valid_rows = np.any(valid, axis=1)
    if np.any(valid_rows):
        filtered = np.where(valid[valid_rows], stacked[valid_rows], -np.inf)
        out[valid_rows] = filtered.max(axis=1)
    return out


def clip_strength(values: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    finite = np.isfinite(values)
    if high <= low:
        return out
    out[finite] = np.clip((values[finite] - low) / (high - low), 0.0, 1.0)
    return out


def descending_strength(values: np.ndarray, low_good: float, high_bad: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    finite = np.isfinite(values)
    if high_bad <= low_good:
        return out
    out[finite] = np.clip((high_bad - values[finite]) / (high_bad - low_good), 0.0, 1.0)
    return out

# tinh toan goc giua 3 diem tren moi frame: 
# diem giua (point_b) duoc xem nhu dinh cua goc, va goc duoc tinh bang arccos cua tich vo huong cua 2 vector point_a->point_b va point_c->point_b chia cho tich do cua 2 vector do
# input la 3 mang 3D voi shape (num_frames, num_landmarks, 3), output la mang 1D voi shape (num_frames,) chua goc tinh duoc tren moi frame, neu khong tinh duoc thi tra ve NaN
# dung de tinh goc khop khuỷu tay va khop goi tren moi frame
def angle_series(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> np.ndarray:
    vec1 = point_a[:, :2] - point_b[:, :2]
    vec2 = point_c[:, :2] - point_b[:, :2]
    denom = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
    out = np.full(point_a.shape[0], np.nan, dtype=np.float32)
    valid = (
        np.all(np.isfinite(vec1), axis=1)
        & np.all(np.isfinite(vec2), axis=1)
        & (denom > 1e-6)
    )
    if np.any(valid):
        cos_angle = np.sum(vec1[valid] * vec2[valid], axis=1) / denom[valid]
        out[valid] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))).astype(np.float32)
    return out


def distance_series(point_a: np.ndarray, point_b: np.ndarray, dims: tuple[int, ...] = (0, 1)) -> np.ndarray:
    diff = point_a[:, dims] - point_b[:, dims]
    out = np.full(point_a.shape[0], np.nan, dtype=np.float32)
    valid = np.all(np.isfinite(diff), axis=1)
    if np.any(valid):
        out[valid] = np.linalg.norm(diff[valid], axis=1).astype(np.float32)
    return out

# values: chuoi feature theo frame 
# person_present: chuoi nhi phan cho biet co nguoi hay khong tren moi frame
# window: kich thuoc cua cua so trung binh dong
# ham nay chia video thanh cac doan lien tuc co nguoi, trong moi doan:
# noi suy cho bi thieu, smooth chuoi
# ghep lai dung vi tri ban dau
# vi du: [10,12,50,13,11] -> [10.5, 18, 19, 15, 12]
def smooth_feature_series(values: np.ndarray, person_present: np.ndarray, window: int = FEATURE_SMOOTH_WINDOW) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for segment in split_present_segments(person_present):
        segment_values = out[segment]
        local_indices = np.arange(segment.size, dtype=np.float32)
        filled = interpolate_series(segment_values, local_indices)
        if not np.any(np.isfinite(filled)):
            continue
        out[segment] = smooth_series(filled, window)
    return out

# diff[t] = values[t] - values[t-1], voi values da duoc interpolate va smooth trong moi doan co nguoi, neu khong tinh duoc thi tra ve NaN
# dung de tinh center_y_velocity (ngoi xuong / dung len), dtheta_left, dtheta_right (tay dang quay theo chieu nao?) tren moi frame
# knee_angle_diff 
# dtheta_left > 0 = tay trai dang quay theo chieu kim dong ho, dtheta_left < 0 = tay trai dang quay nguoc chieu kim dong ho
def diff_feature_series(values: np.ndarray, person_present: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    for segment in split_present_segments(person_present):
        local_values = values[segment]
        local_indices = np.arange(segment.size, dtype=np.float32)
        filled = interpolate_series(local_values, local_indices)
        if not np.any(np.isfinite(filled)):
            continue
        diff = np.diff(filled, prepend=filled[0]).astype(np.float32)
        out[segment] = diff
    return out

# ham tinh toan toc do di chuyen cua mot diem tren moi frame, voi diem duoc bieu dien boi mang 3D voi shape (num_frames, 3), chi tinh toan tren 2D (x,y), neu khong tinh duoc thi tra ve NaN
# dung de tinh toan toc do di chuyen cua tay (wrist) tren moi frame
# left_wrist_speed > 0.1 = tay dang di chuyen nhanh, left_wrist_speed < 0.02 = tay gan nhu dung yen
# tính dx, dy rồi gộp lại thành sqrt(dx² + dy²)
def point_speed_series(points: np.ndarray, person_present: np.ndarray) -> np.ndarray:
    out = np.full(points.shape[0], np.nan, dtype=np.float32)
    for segment in split_present_segments(person_present):
        local_points = points[segment, :2].astype(np.float32, copy=True)
        local_indices = np.arange(segment.size, dtype=np.float32)
        for dim in range(local_points.shape[1]):
            local_points[:, dim] = interpolate_series(local_points[:, dim], local_indices)
        if not np.any(np.isfinite(local_points)):
            continue
        diff = np.diff(local_points, axis=0, prepend=local_points[:1])
        out[segment] = np.linalg.norm(diff, axis=1).astype(np.float32)
    return out

# ham tinh toan goc theta cua tay so voi vai tren moi frame, voi theta duoc tinh bang arctan2 cua theta_left va theta_right
# sau do duoc unwrap de tranh cac giay lat goc dot ngot tu 180 sang -180 do, neu khong tinh duoc thi tra ve NaN
def unwrap_series(values: np.ndarray, person_present: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    for segment in split_present_segments(person_present):
        local_values = values[segment].astype(np.float32, copy=True)
        local_indices = np.arange(segment.size, dtype=np.float32)
        filled = interpolate_series(local_values, local_indices)
        if not np.any(np.isfinite(filled)):
            continue
        out[segment] = np.unwrap(filled).astype(np.float32)
    return out

# trong một cửa sổ quanh frame hiện tại, giá trị dao động rộng bao nhiêu 
def rolling_range(values: np.ndarray, person_present: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    min_valid = max(4, window // 2)
    for segment in split_present_segments(person_present):
        local_values = values[segment]
        for idx in range(segment.size):
            start = max(0, idx - window // 2)
            end = min(segment.size, idx + window // 2 + 1)
            window_values = local_values[start:end]
            valid = np.isfinite(window_values)
            if np.count_nonzero(valid) < min_valid:
                continue
            valid_values = window_values[valid]
            out[segment[idx]] = float(valid_values.max() - valid_values.min())
    return out

# Hàm này tính tương quan trượt giữa hai chuỗi trong một cửa sổ. 
#dtheta_left: tốc độ góc của tay trái
#dtheta_right: tốc độ góc của tay phải
# Nếu hai tay cùng quay đúng nhịp, hai chuỗi này sẽ tương quan cao.
def rolling_corr(x_values: np.ndarray, y_values: np.ndarray, person_present: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x_values.shape[0], np.nan, dtype=np.float32)
    min_valid = max(5, window // 2)
    for segment in split_present_segments(person_present):
        x_local = x_values[segment]
        y_local = y_values[segment]
        for idx in range(segment.size):
            start = max(0, idx - window // 2)
            end = min(segment.size, idx + window // 2 + 1)
            x_window = x_local[start:end]
            y_window = y_local[start:end]
            valid = np.isfinite(x_window) & np.isfinite(y_window)
            if np.count_nonzero(valid) < min_valid:
                continue
            x_valid = x_window[valid]
            y_valid = y_window[valid]
            if np.std(x_valid) < 1e-6 or np.std(y_valid) < 1e-6:
                out[segment[idx]] = 0.0
            else:
                out[segment[idx]] = float(np.corrcoef(x_valid, y_valid)[0, 1])
    return out

# input: normalized pose sequence chua toan bo chuoi pose da norm
# output: DataFrame chua cac feature duoc tinh toan tren moi frame, 
# voi moi dong tuong ung voi 1 frame va cac cot tuong ung voi cac feature khac nhau nhu 
# center_y, shoulder_center_x_norm, left_wrist_above_left_shoulder, left_elbow_angle, 
# left_wrist_speed, theta_left, circle_range_left, arm_sync, v.v.
def build_frame_features(sequence: PoseSequence) -> pd.DataFrame:
    coords = sequence.coords_norm
    visibility = sequence.visibility
    person_present = sequence.person_present
    timestamps_ms = sequence.timestamps_ms
    centers = sequence.centers
    scales = sequence.scales
    # extract landmark quan trong
    nose = coords[:, NOSE_IDX]
    left_shoulder = coords[:, LEFT_SHOULDER_IDX]
    right_shoulder = coords[:, RIGHT_SHOULDER_IDX]
    left_elbow = coords[:, LEFT_ELBOW_IDX]
    right_elbow = coords[:, RIGHT_ELBOW_IDX]
    left_wrist = coords[:, LEFT_WRIST_IDX]
    right_wrist = coords[:, RIGHT_WRIST_IDX]
    left_hip = coords[:, LEFT_HIP_IDX]
    right_hip = coords[:, RIGHT_HIP_IDX]
    left_knee = coords[:, LEFT_KNEE_IDX]
    right_knee = coords[:, RIGHT_KNEE_IDX]
    left_ankle = coords[:, LEFT_ANKLE_IDX]
    right_ankle = coords[:, RIGHT_ANKLE_IDX]

    # tinh diem trung binh cua vai va hong tren moi frame
    shoulder_center = mean_points(coords[:, [LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX]])
    hip_center = mean_points(coords[:, [LEFT_HIP_IDX, RIGHT_HIP_IDX]])

    frame_features: dict[str, np.ndarray] = {
        "frame_idx": np.arange(coords.shape[0], dtype=np.int32),
        "timestamp_ms": timestamps_ms.astype(np.int32),
        "person_present": person_present.astype(np.int8),
        # feature co ban
        "center_y": centers[:, 1].astype(np.float32),
        "scale": scales.astype(np.float32),
        "nose_y_norm": nose[:, 1].astype(np.float32),
        "shoulder_center_x_norm": shoulder_center[:, 0].astype(np.float32),
        "shoulder_center_y_norm": shoulder_center[:, 1].astype(np.float32),

        "left_wrist_above_left_shoulder": (left_shoulder[:, 1] - left_wrist[:, 1]).astype(np.float32),
        "right_wrist_above_right_shoulder": (right_shoulder[:, 1] - right_wrist[:, 1]).astype(np.float32),
        "left_wrist_above_head": (nose[:, 1] - left_wrist[:, 1]).astype(np.float32),
        "right_wrist_above_head": (nose[:, 1] - right_wrist[:, 1]).astype(np.float32),

        "left_elbow_angle": angle_series(left_shoulder, left_elbow, left_wrist),
        "right_elbow_angle": angle_series(right_shoulder, right_elbow, right_wrist),
        "left_knee_angle": angle_series(left_hip, left_knee, left_ankle),
        "right_knee_angle": angle_series(right_hip, right_knee, right_ankle),

        "dist_wrist_wrist": distance_series(left_wrist, right_wrist),
        # for cross_touch, tay cham vao hong doi ben, nen tinh khoang cach tu tay ben nay den hong doi ben kia, neu tay trai cham hong phai thi tinh khoang cach tu tay trai den hong phai, nguoc lai tinh khoang cach tu tay phai den hong trai
        "dist_left_wrist_right_ankle": distance_series(left_wrist, right_ankle),
        "dist_right_wrist_left_ankle": distance_series(right_wrist, left_ankle),
        # hand on hip: tay cham vao hong doi ben, nen tinh khoang cach tu tay ben nay den hong doi ben kia, neu tay trai cham hong phai thi tinh khoang cach tu tay trai den hong phai, nguoc lai tinh khoang cach tu tay phai den hong trai
        "dist_left_wrist_left_hip": distance_series(left_wrist, left_hip),
        "dist_right_wrist_right_hip": distance_series(right_wrist, right_hip),
        # visibility cua tay, neu visibility thap thi co the do sai, nen coi nhu tay khong duoc nhin thay va de gia tri nay cao se giup phat hien on dinh hon cac dong tac co lien quan den tay nhu raise_lr va arm_circle
        "left_visibility": visibility[:, LEFT_WRIST_IDX].astype(np.float32),
        "right_visibility": visibility[:, RIGHT_WRIST_IDX].astype(np.float32),
    }
    # ---- FEATURE TEMPORAL ----

    frame_features["mean_knee_angle"] = rowwise_nanmean(
        [frame_features["left_knee_angle"], frame_features["right_knee_angle"]]
    )

    frame_features["center_y_velocity"] = diff_feature_series(frame_features["center_y"], person_present)
    frame_features["left_wrist_speed"] = point_speed_series(left_wrist, person_present)
    frame_features["right_wrist_speed"] = point_speed_series(right_wrist, person_present)
    frame_features["wrist_speed_mean"] = rowwise_nanmean(
        [frame_features["left_wrist_speed"], frame_features["right_wrist_speed"]]
    )
    # tinh goc quay cua tay so voi vai tren moi frame, neu theta_left > 0 thi tay trai dang quay theo chieu kim dong ho
    # neu theta_left < 0 thi tay trai dang quay nguoc chieu kim dong ho, tuong tu cho tay phai, neu khong tinh duoc thi tra ve NaN
    # v = (dy, dz)
    theta_left_raw = np.arctan2(
        left_wrist[:, 1] - left_shoulder[:, 1],
        left_wrist[:, 2] - left_shoulder[:, 2],
    ).astype(np.float32)
    theta_right_raw = np.arctan2(
        right_wrist[:, 1] - right_shoulder[:, 1],
        right_wrist[:, 2] - right_shoulder[:, 2],
    ).astype(np.float32)
    frame_features["theta_left"] = unwrap_series(theta_left_raw, person_present)
    frame_features["theta_right"] = unwrap_series(theta_right_raw, person_present)
    frame_features["dtheta_left"] = diff_feature_series(frame_features["theta_left"], person_present)
    frame_features["dtheta_right"] = diff_feature_series(frame_features["theta_right"], person_present)
    frame_features["circle_range_left"] = rolling_range(frame_features["theta_left"], person_present, ROLLING_WINDOW)
    frame_features["circle_range_right"] = rolling_range(frame_features["theta_right"], person_present, ROLLING_WINDOW)

    theta_left_xy_raw = np.arctan2(
        left_wrist[:, 1] - left_shoulder[:, 1],
        left_wrist[:, 0] - left_shoulder[:, 0],
    ).astype(np.float32)
    theta_right_xy_raw = np.arctan2(
        right_wrist[:, 1] - right_shoulder[:, 1],
        right_wrist[:, 0] - right_shoulder[:, 0],
    ).astype(np.float32)
    theta_left_xz_raw = np.arctan2(
        left_wrist[:, 0] - left_shoulder[:, 0],
        left_wrist[:, 2] - left_shoulder[:, 2],
    ).astype(np.float32)
    theta_right_xz_raw = np.arctan2(
        right_wrist[:, 0] - right_shoulder[:, 0],
        right_wrist[:, 2] - right_shoulder[:, 2],
    ).astype(np.float32)

    frame_features["theta_left_xy"] = unwrap_series(theta_left_xy_raw, person_present)
    frame_features["theta_right_xy"] = unwrap_series(theta_right_xy_raw, person_present)
    frame_features["theta_left_xz"] = unwrap_series(theta_left_xz_raw, person_present)
    frame_features["theta_right_xz"] = unwrap_series(theta_right_xz_raw, person_present)
    frame_features["circle_range_left_xy"] = rolling_range(frame_features["theta_left_xy"], person_present, ROLLING_WINDOW)
    frame_features["circle_range_right_xy"] = rolling_range(frame_features["theta_right_xy"], person_present, ROLLING_WINDOW)
    frame_features["circle_range_left_xz"] = rolling_range(frame_features["theta_left_xz"], person_present, ROLLING_WINDOW)
    frame_features["circle_range_right_xz"] = rolling_range(frame_features["theta_right_xz"], person_present, ROLLING_WINDOW)
    frame_features["circle_range_left_best"] = rowwise_nanmax(
        [
            frame_features["circle_range_left"],
            frame_features["circle_range_left_xy"],
            frame_features["circle_range_left_xz"],
        ]
    )
    frame_features["circle_range_right_best"] = rowwise_nanmax(
        [
            frame_features["circle_range_right"],
            frame_features["circle_range_right_xy"],
            frame_features["circle_range_right_xz"],
        ]
    )
    frame_features["arm_sync"] = rolling_corr(
        frame_features["dtheta_left"],
        frame_features["dtheta_right"],
        person_present,
        ROLLING_WINDOW,
    )
    # smooth toan bo feature
    for key in (
        "center_y",
        "center_y_velocity",
        "left_wrist_above_left_shoulder",
        "right_wrist_above_right_shoulder",
        "left_wrist_above_head",
        "right_wrist_above_head",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
        "mean_knee_angle",
        "dist_wrist_wrist",
        "dist_left_wrist_right_ankle",
        "dist_right_wrist_left_ankle",
        "dist_left_wrist_left_hip",
        "dist_right_wrist_right_hip",
        "nose_y_norm",
        "shoulder_center_x_norm",
        "shoulder_center_y_norm",
        "left_wrist_speed",
        "right_wrist_speed",
        "wrist_speed_mean",
        "theta_left",
        "theta_right",
        "theta_left_xy",
        "theta_right_xy",
        "theta_left_xz",
        "theta_right_xz",
        "dtheta_left",
        "dtheta_right",
        "circle_range_left",
        "circle_range_right",
        "circle_range_left_xy",
        "circle_range_right_xy",
        "circle_range_left_xz",
        "circle_range_right_xz",
        "circle_range_left_best",
        "circle_range_right_best",
        "arm_sync",
    ):
        frame_features[key] = smooth_feature_series(frame_features[key], person_present)

    # tinh center_y_delta de phat hien viec hip drop tren moi frame, 
    # bang cach lay center_y cua moi frame tru di baseline_center_y, 
    # voi baseline_center_y duoc tinh bang percentiles cua center_y tren cac frame co nguoi, 
    # neu khong tinh duoc thi tra ve NaN
    present_center_y = frame_features["center_y"][person_present == 1]
    baseline_center_y = float(np.nanpercentile(present_center_y, 20)) if present_center_y.size else 0.0
    frame_features["center_y_delta"] = (frame_features["center_y"] - baseline_center_y).astype(np.float32)

    # convert feature thanh strength tu 0 den 1, voi 0 la khong co dong tac, 1 la co dong tac, 
    # va cac gia tri o giua the hien muc do dong tac gan nhu the nao
    left_head_strength = clip_strength(frame_features["left_wrist_above_head"], 0.08, 0.9)
    right_head_strength = clip_strength(frame_features["right_wrist_above_head"], 0.08, 0.9)
    left_shoulder_strength = clip_strength(frame_features["left_wrist_above_left_shoulder"], 0.2, 1.0)
    right_shoulder_strength = clip_strength(frame_features["right_wrist_above_right_shoulder"], 0.2, 1.0)
    elbow_extension_strength = np.maximum(
        clip_strength(frame_features["left_elbow_angle"], 120.0, 175.0),
        clip_strength(frame_features["right_elbow_angle"], 120.0, 175.0),
    )
    # tinh strength cho viec giu than dung thang tren moi frame, bang cach xem xet shoulder_center_x_norm, 
    # neu shoulder_center_x_norm gan 0 thi than dang dung thang, 
    # neu shoulder_center_x_norm xa 0 thi than dang nghiêng sang ben do,
    upright_strength = 1.0 - clip_strength(np.abs(frame_features["shoulder_center_x_norm"]), 0.1, 0.45)
    # tinh strength cho viec tay dang quay tron tren moi frame, 
    # bang cach xem xet circle_range_left va circle_range_right,
    # neu circle_range >= 3.6 thi tay dang quay tron, neu circle_range <= 1.4 thi tay gan nhu khong quay,
    circle_strength = np.minimum(
        clip_strength(frame_features["circle_range_left"], 1.4, 3.6),
        clip_strength(frame_features["circle_range_right"], 1.4, 3.6),
    )
    sync_strength = clip_strength(frame_features["arm_sync"], 0.15, 0.75)
    speed_strength = clip_strength(frame_features["wrist_speed_mean"], 0.02, 0.14)

    # do gap goi
    knee_bend_strength = clip_strength(150.0 - frame_features["mean_knee_angle"], 10.0, 90.0)
    hip_drop_strength = clip_strength(frame_features["center_y_delta"], 0.02, 0.12)

    cross_touch_min = rowwise_nanmin(
        [
            frame_features["dist_left_wrist_right_ankle"],
            frame_features["dist_right_wrist_left_ankle"],
        ]
    )
    frame_features["cross_touch_min"] = cross_touch_min
    cross_strength = descending_strength(cross_touch_min, 0.65, 1.35)

    bend_strength = clip_strength(frame_features["shoulder_center_y_norm"], -0.7, -0.15)
    nose_bend_strength = clip_strength(frame_features["nose_y_norm"], -1.2, -0.15)

    left_hand_on_hip_strength = descending_strength(frame_features["dist_left_wrist_left_hip"], 0.35, 1.15)
    right_hand_on_hip_strength = descending_strength(frame_features["dist_right_wrist_right_hip"], 0.35, 1.15)
    left_raise_strength = clip_strength(frame_features["left_wrist_above_head"], 0.08, 0.9)
    right_raise_strength = clip_strength(frame_features["right_wrist_above_head"], 0.08, 0.9)
    combo_strength = np.maximum(
        np.minimum(left_hand_on_hip_strength, right_raise_strength),
        np.minimum(right_hand_on_hip_strength, left_raise_strength),
    )
    lateral_strength = clip_strength(np.abs(frame_features["shoulder_center_x_norm"]), 0.1, 0.38)
    # build score cho moi action, voi tung feature duoc tinh toan o tren se duoc gan mot trong so 5 action 
    # va duoc tinh toan strength roi cong don theo mot cong thuc co trong tay, 
    # sau do cac strength se duoc cong don voi nhau theo mot he so nhat dinh de ra score tong cho dong tac do, 
    # neu score cao thi dong tac do co kha nang xay ra tren frame do, 
    # neu score thap thi dong tac do khong xay ra tren frame do
    frame_features["action_1_score"] = (
        0.45 * np.maximum(left_head_strength, right_head_strength)
        + 0.20 * np.maximum(left_shoulder_strength, right_shoulder_strength)
        + 0.20 * elbow_extension_strength # tay co gap khong? 
        + 0.15 * upright_strength
    ).astype(np.float32)
    frame_features["action_2_score"] = (
        0.50 * circle_strength
        + 0.30 * sync_strength
        + 0.20 * speed_strength
    ).astype(np.float32)
    frame_features["action_3_score"] = (
        0.65 * knee_bend_strength
        + 0.35 * hip_drop_strength
    ).astype(np.float32)
    frame_features["action_4_score"] = (
        0.60 * cross_strength
        + 0.25 * bend_strength
        + 0.15 * nose_bend_strength
    ).astype(np.float32)
    frame_features["action_5_score"] = (
        0.65 * combo_strength
        + 0.35 * lateral_strength
    ).astype(np.float32)

    for action_id in range(1, 6):
        score_key = f"action_{action_id}_score"
        # smooth score de giam nhieu do nhi phan va giup viec phat hien on dinh hon, 
        # chi smooth tren cac doan co nguoi
        frame_features[score_key] = smooth_feature_series(frame_features[score_key], person_present)
        candidate = (
            (frame_features[score_key] >= ACTION_THRESHOLDS[action_id])
            & (person_present == 1)
        ).astype(np.int8)
        frame_features[f"action_{action_id}_candidate"] = candidate

    rescue_circle_strength = np.minimum(
        clip_strength(frame_features["circle_range_left_best"], ACTION_2_RESCUE_MIN_RANGE, 3.2),
        clip_strength(frame_features["circle_range_right_best"], ACTION_2_RESCUE_MIN_RANGE, 3.2),
    )
    rescue_sync_strength = clip_strength(frame_features["arm_sync"], ACTION_2_RESCUE_MIN_SYNC, 0.95)
    rescue_speed_strength = np.minimum(
        clip_strength(frame_features["left_wrist_speed"], ACTION_2_RESCUE_MIN_SPEED, 0.22),
        clip_strength(frame_features["right_wrist_speed"], ACTION_2_RESCUE_MIN_SPEED, 0.22),
    )
    frame_features["action_2_rescue_score"] = (
        0.50 * rescue_circle_strength
        + 0.30 * rescue_sync_strength
        + 0.20 * rescue_speed_strength
    ).astype(np.float32)
    frame_features["action_2_rescue_score"] = smooth_feature_series(
        frame_features["action_2_rescue_score"],
        person_present,
    )
    frame_features["action_2_rescue_candidate"] = (
        (frame_features["circle_range_left_best"] >= ACTION_2_RESCUE_MIN_RANGE)
        & (frame_features["circle_range_right_best"] >= ACTION_2_RESCUE_MIN_RANGE)
        & (frame_features["arm_sync"] >= ACTION_2_RESCUE_MIN_SYNC)
        & (frame_features["left_wrist_speed"] >= ACTION_2_RESCUE_MIN_SPEED)
        & (frame_features["right_wrist_speed"] >= ACTION_2_RESCUE_MIN_SPEED)
        & (frame_features["action_1_score"] <= ACTION_2_RESCUE_MAX_ACTION1)
        & (frame_features["mean_knee_angle"] >= ACTION_2_RESCUE_MIN_KNEE_ANGLE)
        & (person_present == 1)
    ).astype(np.int8)

    return pd.DataFrame(frame_features)


def find_first_sustained_onset(signal: np.ndarray, start_idx: int, min_run: int) -> tuple[int | None, float]:
    run_length = 0
    run_start = -1

    for idx in range(start_idx, signal.shape[0]):
        if bool(signal[idx]):
            if run_length == 0:
                run_start = idx
            run_length += 1
            if run_length >= min_run:
                confidence = float(min(1.0, signal[run_start:idx + 1].mean()))
                return run_start, confidence
        else:
            run_length = 0
            run_start = -1

    return None, 0.0


def build_score_signal(features_df: pd.DataFrame, action_id: int) -> np.ndarray:
    score = features_df[f"action_{action_id}_score"].to_numpy(dtype=np.float32)
    candidate = features_df[f"action_{action_id}_candidate"].to_numpy(dtype=np.int8).astype(bool)
    signal = np.where(candidate, score, 0.0).astype(np.float32)
    return signal


def build_aux_signal(features_df: pd.DataFrame, score_key: str, candidate_key: str) -> np.ndarray:
    score = features_df[score_key].to_numpy(dtype=np.float32)
    candidate = features_df[candidate_key].to_numpy(dtype=np.int8).astype(bool)
    return np.where(candidate, score, 0.0).astype(np.float32)


def first_present_frame(person_present: np.ndarray) -> int | None:
    present = np.flatnonzero(person_present == 1)
    if present.size == 0:
        return None
    return int(present[0])


def last_present_frame(person_present: np.ndarray) -> int | None:
    present = np.flatnonzero(person_present == 1)
    if present.size == 0:
        return None
    return int(present[-1])

# features_df: có action_i_score, person_present, timestamp_ms
# video_path: duong dan den video, de doc target bits tu ten file
def decode_segments(features_df: pd.DataFrame, video_path: Path) -> list[SegmentRecord]:
    person_present = features_df["person_present"].to_numpy(dtype=np.int8)
    timestamps_ms = features_df["timestamp_ms"].to_numpy(dtype=np.int32)
    # chi lam viec trong vung co nguoi
    first_frame = first_present_frame(person_present)
    last_frame = last_present_frame(person_present)
    target_bits = parse_target_bits(video_path)

    if first_frame is None or last_frame is None:
        return [
            SegmentRecord(
                action_id=action_id,
                action_name=ACTION_NAMES[action_id],
                status="missing",
                missing=True,
                start_frame=None,
                end_frame=None,
                start_ms=None,
                end_ms=None,
                confidence=0.0,
                target_label=target_bits[action_id - 1] if len(target_bits) == 5 else "",
                onset_rule="no_pose",
                offset_rule="no_pose",
            )
            for action_id in range(1, 6)
        ]
    
    # mỗi action có 1 chuỗi score theo frame
    score_signals = {
        action_id: build_score_signal(features_df, action_id)
        for action_id in range(1, 6)
    }
    action_2_rescue_signal = build_aux_signal(
        features_df,
        "action_2_rescue_score",
        "action_2_rescue_candidate",
    )

    detected_onsets: dict[int, DetectionHit] = {}
    search_start = first_frame
    current_action = 1

    while current_action <= 5 and search_start <= last_frame:
        earliest_candidates: list[tuple[int, DetectionHit]] = []
        # tim onset dau tien co score duoc duy tri trong it nhat MIN_RUN_FRAMES frame lien tuc cho tung action tu current_action den 5, 
        # va luu lai cac candidate duoc tim thay voi action_id, onset_frame va confidence (duoc tinh bang mean score trong doan duy tri)
        for action_id in range(current_action, 6):
            selected_hit: DetectionHit | None = None
            onset, confidence = find_first_sustained_onset(
                score_signals[action_id],
                search_start,
                MIN_RUN_FRAMES,
            )
            if onset is not None:
                selected_hit = DetectionHit(
                    onset_frame=onset,
                    confidence=confidence,
                    onset_rule=f"action_{action_id}_score>={ACTION_THRESHOLDS[action_id]:.2f}",
                )

            if action_id == 2 and current_action >= 2:
                rescue_onset, rescue_confidence = find_first_sustained_onset(
                    action_2_rescue_signal,
                    search_start,
                    ACTION_2_RESCUE_MIN_RUN,
                )
                if rescue_onset is not None:
                    rescue_hit = DetectionHit(
                        onset_frame=rescue_onset,
                        confidence=rescue_confidence,
                        onset_rule="action_2_rescue_multiplane",
                        status="uncertain",
                    )
                    if selected_hit is None or rescue_hit.onset_frame < selected_hit.onset_frame:
                        selected_hit = rescue_hit

            if selected_hit is not None:
                earliest_candidates.append((action_id, selected_hit))

        if not earliest_candidates:
            break
        
        # sap xep cac candidate theo onset_frame truoc, neu trung nhau thi theo confidence, 
        # va chon candidate som nhat (va co confidence cao nhat neu cung onset) lam onset cua current_action,
        earliest_candidates.sort(key=lambda item: (item[1].onset_frame, item[0]))
        candidate_action, candidate_hit = earliest_candidates[0]
        current_onset = next((item for item in earliest_candidates if item[0] == current_action), None)

        if current_onset is None:
            detected_onsets[candidate_action] = candidate_hit
            current_action = candidate_action + 1
            search_start = candidate_hit.onset_frame + SEARCH_GAP_FRAMES
            continue

        current_action_id, current_hit = current_onset
        if candidate_action == current_action or current_hit.onset_frame <= candidate_hit.onset_frame + SKIP_LOOKAHEAD_FRAMES:
            detected_onsets[current_action_id] = current_hit
            current_action = current_action_id + 1
            search_start = current_hit.onset_frame + SEARCH_GAP_FRAMES
            continue

        detected_onsets[candidate_action] = candidate_hit
        current_action = candidate_action + 1
        search_start = candidate_hit.onset_frame + SEARCH_GAP_FRAMES

    segments: list[SegmentRecord] = []
    detected_actions = sorted(detected_onsets.items(), key=lambda item: item[1].onset_frame)
    detected_index = {action_id: idx for idx, (action_id, _) in enumerate(detected_actions)}

    for action_id in range(1, 6):
        target_label = target_bits[action_id - 1] if len(target_bits) == 5 else ""
        if action_id not in detected_onsets:
            next_detected = next((det_id for det_id in range(action_id + 1, 6) if det_id in detected_onsets), None)
            onset_rule = "" if next_detected is None else f"skipped_by_action_{next_detected}"
            segments.append(
                SegmentRecord(
                    action_id=action_id,
                    action_name=ACTION_NAMES[action_id],
                    status="missing",
                    missing=True,
                    start_frame=None,
                    end_frame=None,
                    start_ms=None,
                    end_ms=None,
                    confidence=0.0,
                    target_label=target_label,
                    onset_rule=onset_rule,
                    offset_rule="",
                )
            )
            continue

        detection = detected_onsets[action_id]
        onset_frame = detection.onset_frame
        confidence = detection.confidence
        next_detected_action = next((det_id for det_id in range(action_id + 1, 6) if det_id in detected_onsets), None)

        if action_id == 1:
            start_frame = first_frame
        else:
            start_frame = onset_frame

        if next_detected_action is not None:
            end_frame = max(start_frame, detected_onsets[next_detected_action].onset_frame - 1)
            status = detection.status
            offset_rule = f"before_action_{next_detected_action}"
        else:
            end_frame = last_frame
            status = "detected" if action_id == 5 and detection.status == "detected" else "uncertain"
            offset_rule = "video_end"

        segments.append(
            SegmentRecord(
                action_id=action_id,
                action_name=ACTION_NAMES[action_id],
                status=status,
                missing=False,
                start_frame=start_frame,
                end_frame=end_frame,
                start_ms=int(timestamps_ms[start_frame]) if start_frame is not None else None,
                end_ms=int(timestamps_ms[end_frame]) if end_frame is not None else None,
                confidence=round(float(confidence), 4),
                target_label=target_label,
                onset_rule=detection.onset_rule,
                offset_rule=offset_rule,
            )
        )

    return segments

# RENDER QA OVERLAY:
def reconstruct_raw_coords(sequence: PoseSequence) -> np.ndarray:
    raw_coords = np.full_like(sequence.coords_norm, np.nan, dtype=np.float32)
    valid = (
        (sequence.person_present == 1)
        & np.isfinite(sequence.scales)
        & np.all(np.isfinite(sequence.centers), axis=1)
    )
    if not np.any(valid):
        return raw_coords

    raw_coords[valid] = (
        sequence.coords_norm[valid] * sequence.scales[valid, None, None]
        + sequence.centers[valid, None, :]
    )
    return raw_coords


def pixel_points_from_raw_coords(
    raw_coords: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
) -> dict[int, tuple[int, int]]:
    pixel_points: dict[int, tuple[int, int]] = {}
    for landmark_idx in range(NUM_LANDMARKS):
        point = raw_coords[landmark_idx]
        vis = visibility[landmark_idx]
        if np.isfinite(vis) and vis < VISIBILITY_THRESHOLD:
            continue
        if not np.all(np.isfinite(point[:2])):
            continue
        x_coord = float(point[0])
        y_coord = float(point[1])
        if not (0.0 <= x_coord <= 1.0 and 0.0 <= y_coord <= 1.0):
            continue
        pixel_points[landmark_idx] = (
            int(round(x_coord * (width - 1))),
            int(round(y_coord * (height - 1))),
        )
    return pixel_points


def active_segment_for_frame(segments: list[SegmentRecord], frame_idx: int) -> SegmentRecord | None:
    for segment in segments:
        if segment.start_frame is None or segment.end_frame is None:
            continue
        if segment.start_frame <= frame_idx <= segment.end_frame:
            return segment
    return None


def draw_timeline(
    frame,
    segments: list[SegmentRecord],
    frame_idx: int,
    total_frames: int,
) -> None:
    height, width = frame.shape[:2]
    actual_bar_top = height - 62
    actual_bar_height = 16
    legend_top = height - 38
    legend_height = 26

    cv2.rectangle(frame, (16, actual_bar_top), (width - 16, actual_bar_top + actual_bar_height), (28, 28, 28), -1)
    cv2.rectangle(frame, (16, actual_bar_top), (width - 16, actual_bar_top + actual_bar_height), (90, 90, 90), 1)

    for segment in segments:
        color = ACTION_COLORS[segment.action_id]
        if segment.start_frame is None or segment.end_frame is None:
            continue
        start_x = 16 + int(round((segment.start_frame / max(1, total_frames - 1)) * (width - 32)))
        end_x = 16 + int(round((segment.end_frame / max(1, total_frames - 1)) * (width - 32)))
        if segment.status == "uncertain":
            color = tuple(max(40, channel - 90) for channel in color)
        cv2.rectangle(frame, (start_x, actual_bar_top), (max(start_x + 1, end_x), actual_bar_top + actual_bar_height), color, -1)
        cv2.line(frame, (start_x, actual_bar_top), (start_x, actual_bar_top + actual_bar_height), (235, 235, 235), 1, cv2.LINE_AA)

    pointer_x = 16 + int(round((frame_idx / max(1, total_frames - 1)) * (width - 32)))
    cv2.line(frame, (pointer_x, actual_bar_top - 4), (pointer_x, actual_bar_top + actual_bar_height + 4), (255, 255, 255), 2, cv2.LINE_AA)

    slot_width = (width - 32) // 5
    for idx, segment in enumerate(segments):
        left = 16 + idx * slot_width
        right = width - 16 if idx == 4 else left + slot_width - 4
        color = ACTION_COLORS[segment.action_id]
        fill_color = color if not segment.missing else (45, 45, 45)
        cv2.rectangle(frame, (left, legend_top), (right, legend_top + legend_height), fill_color, -1)
        border_color = color if segment.missing else (255, 255, 255)
        cv2.rectangle(frame, (left, legend_top), (right, legend_top + legend_height), border_color, 1)
        status_text = "MISS" if segment.missing else segment.status.upper()
        cv2.putText(
            frame,
            f"A{segment.action_id} {status_text}",
            (left + 6, legend_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def render_segmentation_overlay(
    video_path: Path,
    output_path: Path,
    sequence: PoseSequence,
    segments: list[SegmentRecord],
    width: int,
    height: int,
    fps: float,
) -> None:
    writer = start_ffmpeg_writer(video_path, output_path, width, height, fps)
    if writer.stdin is None:
        raise RuntimeError(f"Cannot create segmentation QA video: {output_path}")

    raw_coords = reconstruct_raw_coords(sequence)
    total_frames = min(sequence.coords_norm.shape[0], int(sequence.timestamps_ms.shape[0]))
    ffmpeg_error = ""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if writer.stdin and not writer.stdin.closed:
            writer.stdin.close()
        writer.wait()
        raise RuntimeError(f"Cannot open video for QA overlay: {video_path}")

    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < total_frames:
            success, frame = cap.read()
            if not success:
                break

            if sequence.person_present[frame_idx] == 1:
                pixel_points = pixel_points_from_raw_coords(
                    raw_coords[frame_idx],
                    sequence.visibility[frame_idx],
                    width,
                    height,
                )
                draw_pose_from_pixel_points(frame, pixel_points)

            active_segment = active_segment_for_frame(segments, frame_idx)
            title = "NO SEGMENT" if active_segment is None else ACTION_DISPLAY_NAMES[active_segment.action_id]
            status = "none" if active_segment is None else active_segment.status
            confidence = 0.0 if active_segment is None else active_segment.confidence
            missing_text = "n/a" if active_segment is None else str(active_segment.missing).lower()

            cv2.putText(
                frame,
                title,
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"status={status} missing={missing_text} conf={confidence:.2f}",
                (20, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"frame={frame_idx}",
                (20, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            draw_timeline(frame, segments, frame_idx, total_frames)
            writer.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        if writer.stdin and not writer.stdin.closed:
            writer.stdin.close()
        ffmpeg_error = writer.stderr.read().decode("utf-8", errors="replace")

    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for segmentation overlay {video_path.name}: {ffmpeg_error.strip()}")


def process_video(video_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    normalized_path, frame_features_path, segments_path, qa_path = resolve_output_paths(video_path, output_dir)
    if not normalized_path.exists():
        raise FileNotFoundError(f"Normalized skeleton not found: {normalized_path}")

    sequence = load_normalized_pose_csv(normalized_path)
    frame_features = build_frame_features(sequence)
    frame_features.to_csv(frame_features_path, index=False)

    segments = decode_segments(frame_features, video_path)
    segments_df = pd.DataFrame([segment.__dict__ for segment in segments])
    segments_df.to_csv(segments_path, index=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    render_segmentation_overlay(video_path, qa_path, sequence, segments, width, height, fps)
    return frame_features_path, segments_path, qa_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    video_paths = collect_video_paths(args)
    total_videos = len(video_paths)

    print(f"Se xu ly {total_videos} video.")
    print(f"Output dir: {output_dir}")

    for index, video_path in enumerate(video_paths, start=1):
        normalized_path, frame_features_path, segments_path, qa_path = resolve_output_paths(video_path, output_dir)
        if not normalized_path.exists():
            print(f"[{index}/{total_videos}] Skip {video_path} (missing normalized pose: {normalized_path})")
            continue

        if not args.force and outputs_ready((normalized_path, frame_features_path, segments_path, qa_path)):
            print(f"[{index}/{total_videos}] Skip {video_path} (segmentation artifacts already exist)")
            print(f"  Frame features: {frame_features_path}")
            print(f"  Segments: {segments_path}")
            print(f"  QA overlay: {qa_path}")
            continue

        print(f"[{index}/{total_videos}] Processing {video_path} ...")
        frame_features_out, segments_out, qa_out = process_video(video_path, output_dir)
        print(f"  Frame features: {frame_features_out}")
        print(f"  Segments: {segments_out}")
        print(f"  QA overlay: {qa_out}")


if __name__ == "__main__":
    main()
