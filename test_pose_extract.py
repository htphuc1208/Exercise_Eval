from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


DEFAULT_MODEL_PATH = Path("pose_landmarker_heavy.task")
DEFAULT_VIDEO_DIR = Path("video_data")
DEFAULT_OUTPUT_DIR = Path("output")
VISIBILITY_THRESHOLD = 0.2
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}
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
            "and overlay videos go to <output-dir>/overlay_video."
        ),
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


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    skeleton_dir = output_dir / "skeleton"
    overlay_dir = output_dir / "overlay_video"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return skeleton_dir, overlay_dir


def build_csv_output_path(video_path: Path, skeleton_dir: Path) -> Path:
    return skeleton_dir / f"{video_path.stem}_pose_keypoints.csv"


def build_overlay_output_path(video_path: Path, overlay_dir: Path) -> Path:
    return overlay_dir / f"{video_path.stem}_pose_overlay.mp4"


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


def write_pose_rows(csv_writer: csv.writer, frame_idx: int, timestamp_ms: int, result) -> list | None:
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        for landmark_idx, landmark in enumerate(landmarks):
            csv_writer.writerow([
                frame_idx,
                timestamp_ms,
                landmark_idx,
                landmark.x,
                landmark.y,
                landmark.z,
                getattr(landmark, "visibility", None),
                1,
            ])
        return landmarks

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
    return None


def process_video(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, int]:
    skeleton_dir, overlay_dir = ensure_output_dirs(output_dir)
    csv_path = build_csv_output_path(video_path, skeleton_dir)
    overlay_path = build_overlay_output_path(video_path, overlay_dir)

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
                    landmarks = write_pose_rows(csv_writer, frame_count, timestamp_ms, result)

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

    return csv_path, overlay_path, frame_count


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
        print(f"[{index}/{total_videos}] Processing {video_path} ...")
        csv_path, overlay_path, frame_count = process_video(video_path, model_path, output_dir)
        print(f"  CSV: {csv_path}")
        print(f"  Overlay: {overlay_path}")
        print(f"  Frames: {frame_count}")


if __name__ == "__main__":
    main()
