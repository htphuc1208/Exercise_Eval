from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


DEFAULT_MODEL_PATH = Path("pose_landmarker_heavy.task")
DEFAULT_VIDEO_DIR = Path("video_data")
DEFAULT_OUTPUT_DIR = Path("output")
VISIBILITY_THRESHOLD = 0.2
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}
SMOOTHING_WINDOW = 5
CENTER_SMOOTHING_WINDOW = 7
NUM_LANDMARKS = 33
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
NORMALIZED_RENDER_SCALE = 0.22
CSV_HEADER = (
    "frame_idx",
    "timestamp_ms",
    "landmark_idx",
    "x",
    "y",
    "z",
    "visibility",
    "person_present",
)
NORMALIZED_CSV_HEADER = (
    "frame_idx",
    "timestamp_ms",
    "landmark_idx",
    "x",
    "y",
    "z",
    "visibility",
    "person_present",
    "center_x",
    "center_y",
    "center_z",
    "scale",
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

# MediaPipe Pose uses 33 landmarks with these fixed skeleton connections.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
    (27, 31), (28, 32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks and render overlay videos."
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
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"MediaPipe model path. Default: {DEFAULT_MODEL_PATH}",
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
            "Root output directory. CSV files go to <output-dir>/skeleton "
            "normalized pose files go to <output-dir>/normalized_skeleton, "
            "overlay videos go to <output-dir>/overlay_video, and normalized "
            "overlay videos go to <output-dir>/normalized_overlay_video."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess videos even if all expected output files already exist.",
    )
    return parser.parse_args()


def list_available_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(
        path for path in video_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.name.endswith("_pose_overlay.mp4")
    )


def list_videos_in_dir(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    return sorted(
        path.resolve() for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.name.endswith("_pose_overlay.mp4")
    )


def resolve_input_paths(input_arg: str, video_dir: Path) -> list[Path]:
    raw_path = Path(input_arg).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(video_dir / raw_path)
        if raw_path.suffix == "":
            candidates.extend([
                video_dir / f"{raw_path.name}.mp4",
                video_dir / raw_path.name,
            ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                raise FileNotFoundError(f"Unsupported video format: {input_arg}")
            return [candidate.resolve()]
        if candidate.exists() and candidate.is_dir():
            video_paths = list_videos_in_dir(candidate)
            if not video_paths:
                raise FileNotFoundError(f"No video files found in directory: {candidate}")
            return video_paths

    raise FileNotFoundError(f"Video or directory not found: {input_arg}")


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


def build_landmarker_options(model_path: Path) -> PoseLandmarkerOptions:
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    skeleton_dir = output_dir / "skeleton"
    normalized_dir = output_dir / "normalized_skeleton"
    overlay_dir = output_dir / "overlay_video"
    normalized_overlay_dir = output_dir / "normalized_overlay_video"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    normalized_overlay_dir.mkdir(parents=True, exist_ok=True)
    return skeleton_dir, normalized_dir, overlay_dir, normalized_overlay_dir


def build_csv_output_path(video_path: Path, skeleton_dir: Path) -> Path:
    return skeleton_dir / f"{video_path.stem}_pose_keypoints.csv"


def build_normalized_csv_output_path(video_path: Path, normalized_dir: Path) -> Path:
    return normalized_dir / f"{video_path.stem}_pose_keypoints_normalized.csv"


def build_overlay_output_path(video_path: Path, overlay_dir: Path) -> Path:
    return overlay_dir / f"{video_path.stem}_pose_overlay.mp4"


def build_normalized_overlay_output_path(video_path: Path, normalized_overlay_dir: Path) -> Path:
    return normalized_overlay_dir / f"{video_path.stem}_pose_normalized_overlay.mp4"


def resolve_output_paths(
    video_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    skeleton_dir, normalized_dir, overlay_dir, normalized_overlay_dir = ensure_output_dirs(output_dir)
    csv_path = build_csv_output_path(video_path, skeleton_dir)
    normalized_csv_path = build_normalized_csv_output_path(video_path, normalized_dir)
    overlay_path = build_overlay_output_path(video_path, overlay_dir)
    normalized_overlay_path = build_normalized_overlay_output_path(video_path, normalized_overlay_dir)
    return csv_path, normalized_csv_path, overlay_path, normalized_overlay_path


def outputs_ready(output_paths: tuple[Path, Path, Path, Path]) -> bool:
    return all(path.exists() for path in output_paths)


def normalized_to_pixel(landmark, width: int, height: int) -> tuple[int, int] | None:
    visibility = getattr(landmark, "visibility", None)
    if visibility is not None and visibility < VISIBILITY_THRESHOLD:
        return None
    if not (0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0):
        return None

    return (
        int(round(landmark.x * (width - 1))),
        int(round(landmark.y * (height - 1))),
    )


def draw_pose_overlay(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    line_thickness = max(2, min(height, width) // 240)
    point_radius = max(3, min(height, width) // 150)

    pixel_points: dict[int, tuple[int, int]] = {}
    for landmark_idx, landmark in enumerate(landmarks):
        point = normalized_to_pixel(landmark, width, height)
        if point is not None:
            pixel_points[landmark_idx] = point

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_point = pixel_points.get(start_idx)
        end_point = pixel_points.get(end_idx)
        if start_point is None or end_point is None:
            continue
        cv2.line(frame, start_point, end_point, (0, 220, 0), line_thickness, cv2.LINE_AA)

    for point in pixel_points.values():
        cv2.circle(frame, point, point_radius, (0, 140, 255), -1, cv2.LINE_AA)


def draw_pose_from_pixel_points(
    frame,
    pixel_points: dict[int, tuple[int, int]],
    line_color: tuple[int, int, int] = (0, 220, 0),
    point_color: tuple[int, int, int] = (0, 140, 255),
) -> None:
    height, width = frame.shape[:2]
    line_thickness = max(2, min(height, width) // 240)
    point_radius = max(3, min(height, width) // 150)

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_point = pixel_points.get(start_idx)
        end_point = pixel_points.get(end_idx)
        if start_point is None or end_point is None:
            continue
        cv2.line(frame, start_point, end_point, line_color, line_thickness, cv2.LINE_AA)

    for point in pixel_points.values():
        cv2.circle(frame, point, point_radius, point_color, -1, cv2.LINE_AA)


def normalized_to_canvas_pixel(
    coord_xyz: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    if not np.all(np.isfinite(coord_xyz[:2])):
        return None

    center_x = width / 2.0
    center_y = height / 2.0
    pixel_scale = min(width, height) * NORMALIZED_RENDER_SCALE

    return (
        int(round(center_x + coord_xyz[0] * pixel_scale)),
        int(round(center_y + coord_xyz[1] * pixel_scale)),
    )


def draw_normalized_reference(frame) -> None:
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    pixel_scale = int(round(min(width, height) * NORMALIZED_RENDER_SCALE))

    frame[:] = (18, 18, 18)
    cv2.line(frame, (0, center_y), (width - 1, center_y), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(frame, (center_x, 0), (center_x, height - 1), (60, 60, 60), 1, cv2.LINE_AA)

    for offset in (-2, -1, 1, 2):
        x = center_x + offset * pixel_scale
        y = center_y + offset * pixel_scale
        if 0 <= x < width:
            cv2.line(frame, (x, center_y - 8), (x, center_y + 8), (80, 80, 80), 1, cv2.LINE_AA)
        if 0 <= y < height:
            cv2.line(frame, (center_x - 8, y), (center_x + 8, y), (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "normalized pose",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


def render_normalized_overlay_video(
    video_path: Path,
    output_path: Path,
    normalized_coords: np.ndarray,
    visibility: np.ndarray,
    person_present: np.ndarray,
    width: int,
    height: int,
    fps: float,
) -> None:
    writer = start_ffmpeg_writer(video_path, output_path, width, height, fps)
    if writer.stdin is None:
        raise RuntimeError(f"Cannot create normalized overlay video: {output_path}")

    ffmpeg_error = ""
    try:
        for frame_idx in range(normalized_coords.shape[0]):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_normalized_reference(frame)

            if person_present[frame_idx] == 1:
                pixel_points: dict[int, tuple[int, int]] = {}
                for landmark_idx in range(NUM_LANDMARKS):
                    vis = visibility[frame_idx, landmark_idx]
                    if np.isfinite(vis) and vis < VISIBILITY_THRESHOLD:
                        continue
                    point = normalized_to_canvas_pixel(normalized_coords[frame_idx, landmark_idx], width, height)
                    if point is not None:
                        pixel_points[landmark_idx] = point

                draw_pose_from_pixel_points(
                    frame,
                    pixel_points,
                    line_color=(90, 255, 90),
                    point_color=(0, 200, 255),
                )
            else:
                cv2.putText(
                    frame,
                    "no pose",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"frame={frame_idx}",
                (20, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            writer.stdin.write(frame.tobytes())
    finally:
        if writer.stdin and not writer.stdin.closed:
            writer.stdin.close()
        ffmpeg_error = writer.stderr.read().decode("utf-8", errors="replace")

    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed for normalized overlay {video_path.name}: {ffmpeg_error.strip()}"
        )


def start_ffmpeg_writer(
    video_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to write a compatible MP4 output.")

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def write_pose_rows(
    csv_writer: csv.writer,
    frame_idx: int,
    timestamp_ms: int,
    result,
) -> tuple[list | None, np.ndarray, np.ndarray, int]:
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        coords = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
        visibility = np.full(NUM_LANDMARKS, np.nan, dtype=np.float32)
        for landmark_idx, landmark in enumerate(landmarks):
            coords[landmark_idx] = (landmark.x, landmark.y, landmark.z)
            visibility_raw = getattr(landmark, "visibility", np.nan)
            visibility_value = np.nan if visibility_raw is None else float(visibility_raw)
            visibility[landmark_idx] = visibility_value
            csv_writer.writerow([
                frame_idx,
                timestamp_ms,
                landmark_idx,
                landmark.x,
                landmark.y,
                landmark.z,
                None if np.isnan(visibility_value) else visibility_value,
                1,
            ])
        return landmarks, coords, visibility, 1

    csv_writer.writerow([
        frame_idx,
        timestamp_ms,
        -1,
        None,
        None,
        None,
        None,
        0,
    ])
    coords = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    visibility = np.full(NUM_LANDMARKS, np.nan, dtype=np.float32)
    return None, coords, visibility, 0


def valid_point_mask(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points), axis=1)


def mean_valid_points(points: np.ndarray) -> np.ndarray:
    valid = valid_point_mask(points)
    if not np.any(valid):
        return np.full(3, np.nan, dtype=np.float32)
    return points[valid].mean(axis=0).astype(np.float32)


def point_distance_xy(point_a: np.ndarray, point_b: np.ndarray) -> float:
    if not np.all(np.isfinite(point_a[:2])) or not np.all(np.isfinite(point_b[:2])):
        return float("nan")
    return float(np.linalg.norm(point_a[:2] - point_b[:2]))


def split_present_segments(person_present: np.ndarray) -> list[np.ndarray]:
    present_indices = np.flatnonzero(person_present == 1)
    if present_indices.size == 0:
        return []
    split_points = np.where(np.diff(present_indices) > 1)[0] + 1
    return [segment for segment in np.split(present_indices, split_points) if segment.size > 0]


def interpolate_series(values: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        return values.copy()
    if np.count_nonzero(valid) == 1:
        filled = np.full_like(values, values[valid][0], dtype=np.float32)
        return filled
    return np.interp(frame_indices, frame_indices[valid], values[valid]).astype(np.float32)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size == 1 or window <= 1:
        return values.astype(np.float32)
    effective_window = min(window, values.size)
    if effective_window % 2 == 0:
        effective_window -= 1
    if effective_window <= 1:
        return values.astype(np.float32)
    kernel = np.full(effective_window, 1.0 / effective_window, dtype=np.float32)
    pad = effective_window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_frame_center(coords: np.ndarray) -> np.ndarray:
    hip_center = mean_valid_points(coords[[LEFT_HIP_IDX, RIGHT_HIP_IDX]])
    if np.all(np.isfinite(hip_center)):
        return hip_center

    shoulder_center = mean_valid_points(coords[[LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX]])
    if np.all(np.isfinite(shoulder_center)):
        return shoulder_center

    return mean_valid_points(coords)


def compute_frame_scale(coords: np.ndarray) -> float:
    shoulder_center = mean_valid_points(coords[[LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX]])
    hip_center = mean_valid_points(coords[[LEFT_HIP_IDX, RIGHT_HIP_IDX]])
    shoulder_width = point_distance_xy(coords[LEFT_SHOULDER_IDX], coords[RIGHT_SHOULDER_IDX])
    torso_length = point_distance_xy(shoulder_center, hip_center)

    candidates = [
        value for value in (shoulder_width, torso_length)
        if np.isfinite(value) and value > 1e-6
    ]
    if not candidates:
        return float("nan")
    return float(max(candidates))


def fill_center_and_scale_gaps(
    centers: np.ndarray,
    scales: np.ndarray,
    person_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    filled_centers = centers.copy()
    filled_scales = scales.copy()

    for segment in split_present_segments(person_present):
        for dim in range(3):
            filled_centers[segment, dim] = interpolate_series(filled_centers[segment, dim], segment)
        filled_scales[segment] = interpolate_series(filled_scales[segment], segment)

    return filled_centers, filled_scales


def stabilize_centers_and_scales(
    centers: np.ndarray,
    scales: np.ndarray,
    person_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stable_centers = centers.copy()
    stable_scales = scales.copy()

    for segment in split_present_segments(person_present):
        local_indices = np.arange(segment.size, dtype=np.float32)

        # Hip/shoulder centers can jitter frame to frame when one side is briefly noisy.
        for dim in range(3):
            filled_center = interpolate_series(stable_centers[segment, dim], local_indices)
            if not np.any(np.isfinite(filled_center)):
                continue
            stable_centers[segment, dim] = smooth_series(filled_center, CENTER_SMOOTHING_WINDOW)

        # Frame-wise 2D body scale collapses during deep bends and cross-touch due to
        # foreshortening. Use one robust scale per continuous pose segment instead.
        filled_scale = interpolate_series(stable_scales[segment], local_indices)
        valid_scale = np.isfinite(filled_scale) & (filled_scale > 1e-6)
        if not np.any(valid_scale):
            continue
        robust_scale = float(np.median(filled_scale[valid_scale]))
        stable_scales[segment] = np.full(segment.size, robust_scale, dtype=np.float32)

    return stable_centers, stable_scales


def normalize_and_smooth_pose_sequence(
    raw_coords: np.ndarray,
    person_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = raw_coords.shape[0]
    centers = np.full((num_frames, 3), np.nan, dtype=np.float32)
    scales = np.full(num_frames, np.nan, dtype=np.float32)

    for frame_idx in range(num_frames):
        if person_present[frame_idx] != 1:
            continue
        centers[frame_idx] = compute_frame_center(raw_coords[frame_idx])
        scales[frame_idx] = compute_frame_scale(raw_coords[frame_idx])

    centers, scales = fill_center_and_scale_gaps(centers, scales, person_present)
    centers, scales = stabilize_centers_and_scales(centers, scales, person_present)

    normalized = np.full_like(raw_coords, np.nan, dtype=np.float32)
    present_segments = split_present_segments(person_present)

    for frame_idx in np.flatnonzero(person_present == 1):
        scale = scales[frame_idx]
        center = centers[frame_idx]
        if not np.isfinite(scale) or scale <= 1e-6 or not np.all(np.isfinite(center)):
            continue
        valid = valid_point_mask(raw_coords[frame_idx])
        if np.any(valid):
            normalized[frame_idx, valid] = (raw_coords[frame_idx, valid] - center) / scale

    smoothed = normalized.copy()
    for segment in present_segments:
        for landmark_idx in range(NUM_LANDMARKS):
            for dim in range(3):
                series = smoothed[segment, landmark_idx, dim]
                filled = interpolate_series(series, segment)
                if not np.any(np.isfinite(filled)):
                    continue
                smoothed[segment, landmark_idx, dim] = smooth_series(filled, SMOOTHING_WINDOW)

    return smoothed, centers, scales


def write_normalized_pose_csv(
    output_path: Path,
    timestamps_ms: list[int],
    normalized_coords: np.ndarray,
    visibility: np.ndarray,
    person_present: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(NORMALIZED_CSV_HEADER)

        for frame_idx, timestamp_ms in enumerate(timestamps_ms):
            if person_present[frame_idx] != 1:
                writer.writerow([
                    frame_idx,
                    timestamp_ms,
                    -1,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                ])
                continue

            center = centers[frame_idx]
            scale = scales[frame_idx]
            center_x = None if not np.isfinite(center[0]) else float(center[0])
            center_y = None if not np.isfinite(center[1]) else float(center[1])
            center_z = None if not np.isfinite(center[2]) else float(center[2])
            scale_value = None if not np.isfinite(scale) else float(scale)

            for landmark_idx in range(NUM_LANDMARKS):
                xyz = normalized_coords[frame_idx, landmark_idx]
                vis = visibility[frame_idx, landmark_idx]
                writer.writerow([
                    frame_idx,
                    timestamp_ms,
                    landmark_idx,
                    None if not np.isfinite(xyz[0]) else float(xyz[0]),
                    None if not np.isfinite(xyz[1]) else float(xyz[1]),
                    None if not np.isfinite(xyz[2]) else float(xyz[2]),
                    None if not np.isfinite(vis) else float(vis),
                    1,
                    center_x,
                    center_y,
                    center_z,
                    scale_value,
                ])


def process_video(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, int]:
    csv_path, normalized_csv_path, overlay_path, normalized_overlay_path = resolve_output_paths(
        video_path,
        output_dir,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_writer = start_ffmpeg_writer(video_path, overlay_path, width, height, fps)
    if ffmpeg_writer.stdin is None:
        cap.release()
        raise RuntimeError(f"Cannot create output video: {overlay_path}")

    frame_count = 0
    ffmpeg_error = ""
    raw_coords_sequence: list[np.ndarray] = []
    visibility_sequence: list[np.ndarray] = []
    person_present_sequence: list[int] = []
    timestamps_ms: list[int] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

        options = build_landmarker_options(model_path)
        with PoseLandmarker.create_from_options(options) as landmarker:
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=frame_rgb,
                    )

                    timestamp_ms = int((frame_count / fps) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    landmarks, coords, visibility, person_present = write_pose_rows(
                        csv_writer,
                        frame_count,
                        timestamp_ms,
                        result,
                    )
                    raw_coords_sequence.append(coords)
                    visibility_sequence.append(visibility)
                    person_present_sequence.append(person_present)
                    timestamps_ms.append(timestamp_ms)

                    if landmarks:
                        draw_pose_overlay(frame, landmarks)

                    cv2.putText(
                        frame,
                        f"frame={frame_count}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    ffmpeg_writer.stdin.write(frame.tobytes())
                    frame_count += 1
            finally:
                cap.release()
                if ffmpeg_writer.stdin and not ffmpeg_writer.stdin.closed:
                    ffmpeg_writer.stdin.close()
                ffmpeg_error = ffmpeg_writer.stderr.read().decode("utf-8", errors="replace")

    return_code = ffmpeg_writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path.name}: {ffmpeg_error.strip()}")

    raw_coords = np.stack(raw_coords_sequence, axis=0) if raw_coords_sequence else np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)
    visibility = np.stack(visibility_sequence, axis=0) if visibility_sequence else np.empty((0, NUM_LANDMARKS), dtype=np.float32)
    person_present = np.asarray(person_present_sequence, dtype=np.int8)
    normalized_coords, centers, scales = normalize_and_smooth_pose_sequence(raw_coords, person_present)
    write_normalized_pose_csv(
        normalized_csv_path,
        timestamps_ms,
        normalized_coords,
        visibility,
        person_present,
        centers,
        scales,
    )
    render_normalized_overlay_video(
        video_path,
        normalized_overlay_path,
        normalized_coords,
        visibility,
        person_present,
        width,
        height,
        fps,
    )

    return csv_path, normalized_csv_path, overlay_path, normalized_overlay_path, frame_count


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_paths = collect_video_paths(args)
    total_videos = len(video_paths)
    print(f"Se xu ly {total_videos} video.")
    print(f"Output dir: {output_dir}")

    for index, video_path in enumerate(video_paths, start=1):
        output_paths = resolve_output_paths(video_path, output_dir)
        if not args.force and outputs_ready(output_paths):
            csv_path, normalized_csv_path, overlay_path, normalized_overlay_path = output_paths
            print(f"[{index}/{total_videos}] Skip {video_path} (outputs already exist)")
            print(f"  Raw CSV: {csv_path}")
            print(f"  Normalized CSV: {normalized_csv_path}")
            print(f"  Overlay: {overlay_path}")
            print(f"  Normalized Overlay: {normalized_overlay_path}")
            continue

        print(f"[{index}/{total_videos}] Processing {video_path} ...")
        csv_path, normalized_csv_path, overlay_path, normalized_overlay_path, frame_count = process_video(
            video_path,
            model_path,
            output_dir,
        )
        print(f"  Raw CSV: {csv_path}")
        print(f"  Normalized CSV: {normalized_csv_path}")
        print(f"  Overlay: {overlay_path}")
        print(f"  Normalized Overlay: {normalized_overlay_path}")
        print(f"  Frames: {frame_count}")


if __name__ == "__main__":
    main()
