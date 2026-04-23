from __future__ import annotations

import argparse
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from action_training_common import (
    DEFAULT_MANUAL_SEGMENTS_DIRNAME,
    DEFAULT_OUTPUT_DIR,
    build_segment_csv_path,
    ensure_segment_dirs,
    parse_boolish,
)
from segment_pose_routine import ACTION_COLORS, ACTION_DISPLAY_NAMES, ACTION_NAMES, parse_target_bits
from test_pose_extract import DEFAULT_VIDEO_DIR, collect_video_paths


WINDOW_NAME = "Manual Segment Labeler"
STATUS_COLORS = {
    "detected": (46, 204, 113),
    "uncertain": (0, 179, 255),
    "missing": (80, 80, 80),
}
CANVAS_BG = (18, 18, 18)
PANEL_BG = (28, 28, 28)
TEXT_COLOR = (235, 235, 235)
MUTED_TEXT_COLOR = (170, 170, 170)
WARNING_COLOR = (60, 120, 255)
CURSOR_COLOR = (0, 210, 255)
AUTO_GUIDE_COLOR = (120, 120, 120)
MAX_VIDEO_DISPLAY_W = 920
MAX_VIDEO_DISPLAY_H = 700
RIGHT_PANEL_W = 380
TIMELINE_H = 220
PADDING = 16
TRACKBAR_NAME = "Frame"
ARROW_LEFT = 2424832
ARROW_RIGHT = 2555904


@dataclass
class SegmentState:
    action_id: int
    action_name: str
    status: str
    missing: bool
    start_frame: int | None
    end_frame: int | None
    confidence: float | None
    target_label: str
    onset_rule: str
    offset_rule: str
    segment_source: str


class FrameCache:
    def __init__(self, max_items: int = 160) -> None:
        self.max_items = max_items
        self._items: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def get(self, source_name: str, frame_idx: int) -> np.ndarray | None:
        key = (source_name, frame_idx)
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value.copy()

    def put(self, source_name: str, frame_idx: int, frame: np.ndarray) -> None:
        key = (source_name, frame_idx)
        self._items[key] = frame.copy()
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


class ManualSegmentEditor:
    def __init__(self, video_path: Path, output_dir: Path, *, video_index: int, video_total: int) -> None:
        self.video_path = video_path.resolve()
        self.video_id = self.video_path.stem
        self.output_dir = output_dir.resolve()
        self.video_index = video_index
        self.video_total = video_total
        self.auto_segments_dir, self.manual_segments_dir = ensure_segment_dirs(self.output_dir)
        self.auto_segments_path = build_segment_csv_path(self.video_id, self.auto_segments_dir)
        self.manual_segments_path = build_segment_csv_path(self.video_id, self.manual_segments_dir)
        self.overlay_video_path = self.output_dir / "overlay_video" / f"{self.video_id}_pose_overlay.mp4"

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Khong mo duoc video: {self.video_path}")

        self.overlay_capture: cv2.VideoCapture | None = None
        if self.overlay_video_path.exists():
            overlay_capture = cv2.VideoCapture(str(self.overlay_video_path))
            if overlay_capture.isOpened():
                self.overlay_capture = overlay_capture
            else:
                overlay_capture.release()

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        scale = min(MAX_VIDEO_DISPLAY_W / max(1, self.frame_width), MAX_VIDEO_DISPLAY_H / max(1, self.frame_height))
        scale = min(scale, 1.0)
        self.display_width = max(320, int(round(self.frame_width * scale)))
        self.display_height = max(240, int(round(self.frame_height * scale)))
        self.canvas_width = self.display_width + RIGHT_PANEL_W + PADDING * 3
        self.canvas_height = self.display_height + TIMELINE_H + PADDING * 3

        self.frame_cache = FrameCache()
        self._trabar_sync = False
        self.playing = False
        self.show_auto_guide = True
        self.show_overlay_video = self.overlay_capture is not None
        self.current_frame = 0
        self.selected_action = 1
        self.dirty = False
        self.last_tick = time.perf_counter()
        self.message = ""
        self.message_deadline = 0.0
        self.window_created = False
        self.timeline_area: tuple[int, int, int, int] = (0, 0, 0, 0)

        self.auto_segments = self._load_segment_map(self.auto_segments_path, source_name="segments")
        self.segments = self._load_segment_map(
            self.manual_segments_path if self.manual_segments_path.exists() else self.auto_segments_path,
            source_name="manual_segments" if self.manual_segments_path.exists() else "segments",
        )
        self.selected_action = self._choose_initial_action()
        initial_segment = self.segments[self.selected_action]
        self.current_frame = 0 if initial_segment.start_frame is None else initial_segment.start_frame

    def _choose_initial_action(self) -> int:
        for action_id in range(1, 6):
            segment = self.segments[action_id]
            if segment.status == "missing":
                return action_id
            if segment.start_frame is None or segment.end_frame is None:
                return action_id
        return 1

    def _default_segment(self, action_id: int) -> SegmentState:
        target_bits = parse_target_bits(self.video_path)
        return SegmentState(
            action_id=action_id,
            action_name=ACTION_NAMES[action_id],
            status="detected",
            missing=False,
            start_frame=None,
            end_frame=None,
            confidence=None,
            target_label=target_bits[action_id - 1] if len(target_bits) == 5 else "",
            onset_rule="manual",
            offset_rule="manual",
            segment_source=DEFAULT_MANUAL_SEGMENTS_DIRNAME,
        )

    def _load_segment_map(self, csv_path: Path, *, source_name: str) -> dict[int, SegmentState]:
        segment_map = {action_id: self._default_segment(action_id) for action_id in range(1, 6)}
        if not csv_path.exists():
            return segment_map

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            action_id = int(row["action_id"])
            if action_id not in segment_map:
                continue
            segment_map[action_id] = SegmentState(
                action_id=action_id,
                action_name=str(row.get("action_name", ACTION_NAMES[action_id])),
                status=str(row.get("status", "detected") or "detected"),
                missing=parse_boolish(row.get("missing", False)),
                start_frame=None if pd.isna(row.get("start_frame")) else int(row["start_frame"]),
                end_frame=None if pd.isna(row.get("end_frame")) else int(row["end_frame"]),
                confidence=None if pd.isna(row.get("confidence")) else float(row["confidence"]),
                target_label="" if pd.isna(row.get("target_label")) else str(row.get("target_label")),
                onset_rule=str(row.get("onset_rule", "manual")),
                offset_rule=str(row.get("offset_rule", "manual")),
                segment_source=str(row.get("segment_source", source_name)),
            )
            if segment_map[action_id].status == "missing":
                segment_map[action_id].missing = True
        return segment_map

    def _capture_for_source(self) -> tuple[str, cv2.VideoCapture]:
        if self.show_overlay_video and self.overlay_capture is not None:
            return "overlay", self.overlay_capture
        return "raw", self.capture

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        source_name, capture = self._capture_for_source()
        cached = self.frame_cache.get(source_name, frame_idx)
        if cached is not None:
            return cached

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.frame_cache.put(source_name, frame_idx, frame)
        return frame.copy()

    def _seek_to(self, frame_idx: int) -> None:
        self.current_frame = int(np.clip(frame_idx, 0, self.total_frames - 1))

    def _sync_trackbar(self) -> None:
        if not self.window_created:
            return
        self._trabar_sync = True
        cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, int(self.current_frame))
        self._trabar_sync = False

    def _set_message(self, text: str, duration_sec: float = 2.5) -> None:
        self.message = text
        self.message_deadline = time.perf_counter() + duration_sec

    def _status_color(self, status: str) -> tuple[int, int, int]:
        return STATUS_COLORS.get(status, TEXT_COLOR)

    def _toggle_missing(self) -> None:
        segment = self.segments[self.selected_action]
        if segment.status == "missing":
            segment.status = "detected"
            segment.missing = False
        else:
            segment.status = "missing"
            segment.missing = True
            segment.start_frame = None
            segment.end_frame = None
        segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME
        self.dirty = True
        self._set_message(f"A{segment.action_id} status -> {segment.status}")

    def _toggle_uncertain(self) -> None:
        segment = self.segments[self.selected_action]
        if segment.status == "missing":
            segment.status = "detected"
            segment.missing = False
        else:
            segment.status = "detected" if segment.status == "uncertain" else "uncertain"
            segment.missing = False
        segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME
        self.dirty = True
        self._set_message(f"A{segment.action_id} status -> {segment.status}")

    def _restore_selected_from_auto(self) -> None:
        auto_segment = self.auto_segments[self.selected_action]
        self.segments[self.selected_action] = SegmentState(
            action_id=auto_segment.action_id,
            action_name=auto_segment.action_name,
            status=auto_segment.status,
            missing=auto_segment.missing,
            start_frame=auto_segment.start_frame,
            end_frame=auto_segment.end_frame,
            confidence=auto_segment.confidence,
            target_label=auto_segment.target_label,
            onset_rule=auto_segment.onset_rule,
            offset_rule=auto_segment.offset_rule,
            segment_source=DEFAULT_MANUAL_SEGMENTS_DIRNAME,
        )
        self.dirty = True
        self._set_message(f"A{self.selected_action} restored from auto segments")

    def _clear_selected_bounds(self) -> None:
        segment = self.segments[self.selected_action]
        segment.start_frame = None
        segment.end_frame = None
        if segment.status == "missing":
            segment.missing = True
        else:
            segment.missing = False
        segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME
        self.dirty = True
        self._set_message(f"A{self.selected_action} bounds cleared")

    def _set_selected_start(self, frame_idx: int) -> None:
        segment = self.segments[self.selected_action]
        frame_idx = int(np.clip(frame_idx, 0, self.total_frames - 1))
        segment.start_frame = frame_idx
        if segment.end_frame is not None and segment.end_frame < frame_idx:
            segment.end_frame = frame_idx
        segment.status = "detected" if segment.status == "missing" else segment.status
        segment.missing = False
        segment.onset_rule = "manual"
        segment.offset_rule = "manual"
        segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME

        if self.selected_action > 1:
            prev_segment = self.segments[self.selected_action - 1]
            if prev_segment.status != "missing" and (prev_segment.end_frame is None or prev_segment.end_frame >= frame_idx):
                prev_end = max(0, frame_idx - 1)
                prev_segment.end_frame = prev_end if prev_segment.start_frame is None else max(prev_segment.start_frame, prev_end)
                prev_segment.offset_rule = "manual"
                prev_segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME

        self.dirty = True
        self._set_message(f"A{self.selected_action} start = {frame_idx}")

    def _set_selected_end(self, frame_idx: int) -> None:
        segment = self.segments[self.selected_action]
        frame_idx = int(np.clip(frame_idx, 0, self.total_frames - 1))
        segment.end_frame = frame_idx
        if segment.start_frame is not None and segment.start_frame > frame_idx:
            segment.start_frame = frame_idx
        segment.status = "detected" if segment.status == "missing" else segment.status
        segment.missing = False
        segment.onset_rule = "manual"
        segment.offset_rule = "manual"
        segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME

        if self.selected_action < 5:
            next_segment = self.segments[self.selected_action + 1]
            if next_segment.status != "missing" and (next_segment.start_frame is None or next_segment.start_frame <= frame_idx):
                next_start = min(self.total_frames - 1, frame_idx + 1)
                next_segment.start_frame = next_start if next_segment.end_frame is None else min(next_segment.end_frame, next_start)
                next_segment.onset_rule = "manual"
                next_segment.segment_source = DEFAULT_MANUAL_SEGMENTS_DIRNAME

        self.dirty = True
        self._set_message(f"A{self.selected_action} end = {frame_idx}")

    def _jump_boundary(self, direction: int) -> None:
        boundaries: list[int] = []
        for segment in self.segments.values():
            if segment.start_frame is not None:
                boundaries.append(segment.start_frame)
            if segment.end_frame is not None:
                boundaries.append(segment.end_frame)
        if not boundaries:
            return
        boundaries = sorted(set(boundaries))
        if direction > 0:
            for boundary in boundaries:
                if boundary > self.current_frame:
                    self._seek_to(boundary)
                    return
            self._seek_to(boundaries[-1])
        else:
            for boundary in reversed(boundaries):
                if boundary < self.current_frame:
                    self._seek_to(boundary)
                    return
            self._seek_to(boundaries[0])

    def _segment_at_frame(self, frame_idx: int, *, use_auto: bool = False) -> SegmentState | None:
        segment_map = self.auto_segments if use_auto else self.segments
        for action_id in range(1, 6):
            segment = segment_map[action_id]
            if segment.start_frame is None or segment.end_frame is None:
                continue
            if segment.start_frame <= frame_idx <= segment.end_frame:
                return segment
        return None

    def _validate_segments(self) -> list[str]:
        issues: list[str] = []
        last_end = -1
        for action_id in range(1, 6):
            segment = self.segments[action_id]
            if segment.status == "missing":
                continue
            if segment.start_frame is None or segment.end_frame is None:
                issues.append(f"A{action_id}: missing bounds")
                continue
            if segment.start_frame < 0 or segment.end_frame >= self.total_frames:
                issues.append(f"A{action_id}: out of frame range")
            if segment.end_frame < segment.start_frame:
                issues.append(f"A{action_id}: end < start")
            if segment.start_frame <= last_end:
                issues.append(f"A{action_id}: overlaps previous")
            last_end = max(last_end, segment.end_frame)
        return issues

    def _segment_to_row(self, segment: SegmentState) -> dict[str, object]:
        missing = segment.status == "missing" or segment.missing
        start_frame = "" if segment.start_frame is None or missing else int(segment.start_frame)
        end_frame = "" if segment.end_frame is None or missing else int(segment.end_frame)
        start_ms = "" if start_frame == "" else int(round((float(start_frame) / max(self.fps, 1e-6)) * 1000.0))
        end_ms = "" if end_frame == "" else int(round((float(end_frame) / max(self.fps, 1e-6)) * 1000.0))
        return {
            "action_id": segment.action_id,
            "action_name": segment.action_name,
            "status": "missing" if missing else segment.status,
            "missing": bool(missing),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": "" if segment.confidence is None else float(segment.confidence),
            "target_label": segment.target_label,
            "onset_rule": "manual",
            "offset_rule": "manual",
            "segment_source": DEFAULT_MANUAL_SEGMENTS_DIRNAME,
        }

    def save_segments(self) -> Path:
        rows = [self._segment_to_row(self.segments[action_id]) for action_id in range(1, 6)]
        pd.DataFrame(rows).to_csv(self.manual_segments_path, index=False)
        self.dirty = False
        issues = self._validate_segments()
        if issues:
            self._set_message(f"Saved with {len(issues)} validation issue(s)")
        else:
            self._set_message(f"Saved {self.manual_segments_path.name}")
        return self.manual_segments_path

    def _draw_timeline(self, canvas: np.ndarray) -> None:
        left = PADDING
        top = self.display_height + PADDING * 2
        width = self.canvas_width - PADDING * 2
        height = TIMELINE_H - PADDING
        self.timeline_area = (left, top, width, height)

        cv2.rectangle(canvas, (left, top), (left + width, top + height), PANEL_BG, -1)
        cv2.putText(canvas, "Timeline: click to seek | Shift+click set start | Ctrl+click set end", (left + 12, top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED_TEXT_COLOR, 1, cv2.LINE_AA)

        label_w = 130
        bar_left = left + label_w
        bar_right = left + width - 12
        row_h = 28
        row_gap = 8
        row_top = top + 42

        current_x = bar_left + int(round((self.current_frame / max(1, self.total_frames - 1)) * (bar_right - bar_left)))

        for idx, action_id in enumerate(range(1, 6)):
            segment = self.segments[action_id]
            auto_segment = self.auto_segments[action_id]
            y = row_top + idx * (row_h + row_gap)
            selected = action_id == self.selected_action
            row_bg = (45, 45, 45) if selected else (32, 32, 32)
            cv2.rectangle(canvas, (left + 8, y), (left + width - 8, y + row_h), row_bg, -1)

            label_color = ACTION_COLORS[action_id]
            cv2.putText(canvas, ACTION_DISPLAY_NAMES[action_id], (left + 14, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1, cv2.LINE_AA)

            cv2.rectangle(canvas, (bar_left, y + 4), (bar_right, y + row_h - 4), (55, 55, 55), -1)

            if self.show_auto_guide and auto_segment.start_frame is not None and auto_segment.end_frame is not None:
                auto_start_x = bar_left + int(round((auto_segment.start_frame / max(1, self.total_frames - 1)) * (bar_right - bar_left)))
                auto_end_x = bar_left + int(round((auto_segment.end_frame / max(1, self.total_frames - 1)) * (bar_right - bar_left)))
                cv2.rectangle(canvas, (auto_start_x, y + 8), (max(auto_start_x + 1, auto_end_x), y + row_h - 8), AUTO_GUIDE_COLOR, 1)

            if segment.start_frame is not None and segment.end_frame is not None and segment.status != "missing":
                seg_start_x = bar_left + int(round((segment.start_frame / max(1, self.total_frames - 1)) * (bar_right - bar_left)))
                seg_end_x = bar_left + int(round((segment.end_frame / max(1, self.total_frames - 1)) * (bar_right - bar_left)))
                cv2.rectangle(canvas, (seg_start_x, y + 6), (max(seg_start_x + 1, seg_end_x), y + row_h - 6), self._status_color(segment.status), -1)
                cv2.rectangle(canvas, (seg_start_x, y + 6), (max(seg_start_x + 1, seg_end_x), y + row_h - 6), (230, 230, 230), 1)
            elif segment.status == "missing":
                cv2.putText(canvas, "missing", (bar_left + 8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED_TEXT_COLOR, 1, cv2.LINE_AA)

            cv2.line(canvas, (current_x, y + 2), (current_x, y + row_h - 2), CURSOR_COLOR, 2)
            if selected:
                cv2.rectangle(canvas, (left + 8, y), (left + width - 8, y + row_h), (230, 230, 230), 1)

    def _draw_side_panel(self, canvas: np.ndarray) -> None:
        left = self.display_width + PADDING * 2
        top = PADDING
        width = RIGHT_PANEL_W
        bottom = self.display_height + PADDING
        cv2.rectangle(canvas, (left, top), (left + width, bottom), PANEL_BG, -1)

        current_time_ms = int(round((self.current_frame / max(self.fps, 1e-6)) * 1000.0))
        active_segment = self._segment_at_frame(self.current_frame)
        auto_active_segment = self._segment_at_frame(self.current_frame, use_auto=True)
        issues = self._validate_segments()
        selected_segment = self.segments[self.selected_action]
        auto_selected_segment = self.auto_segments[self.selected_action]

        y = top + 28
        def put(line: str, color: tuple[int, int, int] = TEXT_COLOR, scale: float = 0.55, dy: int = 24) -> None:
            nonlocal y
            cv2.putText(canvas, line, (left + 14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += dy

        put(f"[{self.video_index}/{self.video_total}] {self.video_id}", scale=0.62)
        put(f"Frame {self.current_frame + 1}/{self.total_frames}  |  {current_time_ms} ms", color=MUTED_TEXT_COLOR)
        source_text = "pose overlay" if self.show_overlay_video and self.overlay_capture is not None else "raw video"
        put(f"View: {source_text}", color=MUTED_TEXT_COLOR)
        put(f"Dirty: {'YES' if self.dirty else 'no'}", color=WARNING_COLOR if self.dirty else MUTED_TEXT_COLOR)

        if active_segment is not None:
            put(f"At frame: A{active_segment.action_id} {active_segment.status}", color=self._status_color(active_segment.status))
        else:
            put("At frame: no manual slot", color=MUTED_TEXT_COLOR)
        if auto_active_segment is not None:
            put(f"Auto ref: A{auto_active_segment.action_id} {auto_active_segment.status}", color=AUTO_GUIDE_COLOR)
        else:
            put("Auto ref: none", color=AUTO_GUIDE_COLOR)

        y += 6
        put(f"Selected: A{selected_segment.action_id} {ACTION_DISPLAY_NAMES[selected_segment.action_id]}", color=ACTION_COLORS[selected_segment.action_id], scale=0.6)
        put(f"Status: {selected_segment.status}", color=self._status_color(selected_segment.status))
        put(f"Manual: {selected_segment.start_frame} -> {selected_segment.end_frame}", color=TEXT_COLOR)
        put(f"Auto:   {auto_selected_segment.start_frame} -> {auto_selected_segment.end_frame}", color=AUTO_GUIDE_COLOR)
        put(f"Target label: {selected_segment.target_label or '-'}", color=MUTED_TEXT_COLOR)

        y += 6
        put("Keys", color=(240, 240, 240), scale=0.6)
        for line in (
            "Space play/pause",
            "A/D or <-/->: +/-1 frame",
            "J/L: +/-5   U/O: +/-30",
            "1-5 select action",
            "S set start   E set end",
            "X clear bounds",
            "M toggle missing",
            "C toggle detected/uncertain",
            "V toggle raw/overlay",
            "T toggle auto guide",
            "R restore selected from auto",
            "B/N prev/next boundary",
            "Enter save + next",
            "Q save + quit   Esc discard quit",
        ):
            put(line, color=MUTED_TEXT_COLOR, scale=0.48, dy=20)

        y += 8
        put(f"Validation issues: {len(issues)}", color=WARNING_COLOR if issues else (46, 204, 113), scale=0.58)
        for issue in issues[:5]:
            put(f"- {issue}", color=WARNING_COLOR, scale=0.46, dy=18)

    def _render_canvas(self) -> np.ndarray:
        canvas = np.full((self.canvas_height, self.canvas_width, 3), CANVAS_BG, dtype=np.uint8)
        frame = self._read_frame(self.current_frame)
        frame_resized = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        canvas[PADDING : PADDING + self.display_height, PADDING : PADDING + self.display_width] = frame_resized

        current_time_ms = int(round((self.current_frame / max(self.fps, 1e-6)) * 1000.0))
        current_segment = self._segment_at_frame(self.current_frame)
        title = "No segment" if current_segment is None else ACTION_DISPLAY_NAMES[current_segment.action_id]
        title_color = TEXT_COLOR if current_segment is None else self._status_color(current_segment.status)
        cv2.putText(canvas, title, (PADDING + 12, PADDING + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, title_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"frame={self.current_frame}  t={current_time_ms}ms", (PADDING + 12, PADDING + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, TEXT_COLOR, 1, cv2.LINE_AA)

        if self.message and time.perf_counter() <= self.message_deadline:
            cv2.putText(
                canvas,
                self.message,
                (PADDING + 12, PADDING + self.display_height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                WARNING_COLOR,
                2,
                cv2.LINE_AA,
            )

        self._draw_side_panel(canvas)
        self._draw_timeline(canvas)
        return canvas

    def _handle_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        left, top, width, height = self.timeline_area
        if not (left <= x <= left + width and top <= y <= top + height):
            return

        label_w = 130
        bar_left = left + label_w
        bar_right = left + width - 12
        row_h = 28
        row_gap = 8
        row_top = top + 42
        if y < row_top:
            if x >= bar_left:
                ratio = np.clip((x - bar_left) / max(1, bar_right - bar_left), 0.0, 1.0)
                self._seek_to(int(round(ratio * (self.total_frames - 1))))
            return

        row_idx = int((y - row_top) // (row_h + row_gap))
        if not (0 <= row_idx < 5):
            return
        action_id = row_idx + 1
        self.selected_action = action_id

        if x >= bar_left:
            ratio = np.clip((x - bar_left) / max(1, bar_right - bar_left), 0.0, 1.0)
            clicked_frame = int(round(ratio * (self.total_frames - 1)))
            self._seek_to(clicked_frame)
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self._set_selected_start(clicked_frame)
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                self._set_selected_end(clicked_frame)

    def _on_trackbar(self, position: int) -> None:
        if self._trabar_sync:
            return
        self._seek_to(position)

    def _handle_key(self, key: int) -> str | None:
        if key < 0:
            return None

        if key in {13, 10}:
            self.save_segments()
            return "next"
        if key == 27:
            return "discard_quit"
        if key in {ord("q"), ord("Q")}:
            self.save_segments()
            return "save_quit"
        if key == ord(" "):
            self.playing = not self.playing
            return None
        if key in {ord("a"), ord("A"), ARROW_LEFT}:
            self._seek_to(self.current_frame - 1)
            return None
        if key in {ord("d"), ord("D"), ARROW_RIGHT}:
            self._seek_to(self.current_frame + 1)
            return None
        if key in {ord("j"), ord("J")}:
            self._seek_to(self.current_frame - 5)
            return None
        if key in {ord("l"), ord("L")}:
            self._seek_to(self.current_frame + 5)
            return None
        if key in {ord("u"), ord("U")}:
            self._seek_to(self.current_frame - 30)
            return None
        if key in {ord("o"), ord("O")}:
            self._seek_to(self.current_frame + 30)
            return None
        if key in {ord("s"), ord("S")}:
            self._set_selected_start(self.current_frame)
            return None
        if key in {ord("e"), ord("E")}:
            self._set_selected_end(self.current_frame)
            return None
        if key in {ord("x"), ord("X")}:
            self._clear_selected_bounds()
            return None
        if key in {ord("m"), ord("M")}:
            self._toggle_missing()
            return None
        if key in {ord("c"), ord("C")}:
            self._toggle_uncertain()
            return None
        if key in {ord("v"), ord("V")} and self.overlay_capture is not None:
            self.show_overlay_video = not self.show_overlay_video
            source_text = "pose overlay" if self.show_overlay_video else "raw video"
            self._set_message(f"View -> {source_text}")
            return None
        if key in {ord("t"), ord("T")}:
            self.show_auto_guide = not self.show_auto_guide
            self._set_message(f"Auto guide -> {'on' if self.show_auto_guide else 'off'}")
            return None
        if key in {ord("r"), ord("R")}:
            self._restore_selected_from_auto()
            return None
        if key in {ord("b"), ord("B")}:
            self._jump_boundary(-1)
            return None
        if key in {ord("n"), ord("N")}:
            self._jump_boundary(1)
            return None
        if ord("1") <= key <= ord("5"):
            self.selected_action = int(chr(key))
            self._set_message(f"Selected A{self.selected_action}")
            return None
        return None

    def _tick_playback(self) -> None:
        if not self.playing:
            self.last_tick = time.perf_counter()
            return

        now = time.perf_counter()
        elapsed = now - self.last_tick
        frame_interval = 1.0 / max(self.fps, 1.0)
        if elapsed < frame_interval:
            return
        steps = max(1, int(elapsed / frame_interval))
        self._seek_to(self.current_frame + steps)
        self.last_tick = now
        if self.current_frame >= self.total_frames - 1:
            self.playing = False

    def run(self) -> str:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(self.canvas_width, 1500), min(self.canvas_height, 1000))
        cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, _userdata=None: self._handle_mouse(event, x, y, flags))
        cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, int(self.current_frame), max(1, self.total_frames - 1), self._on_trackbar)
        self.window_created = True

        while True:
            self._tick_playback()
            self._sync_trackbar()
            canvas = self._render_canvas()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(15)
            result = self._handle_key(key)
            if result is not None:
                cv2.destroyWindow(WINDOW_NAME)
                self.capture.release()
                if self.overlay_capture is not None:
                    self.overlay_capture.release()
                self.window_created = False
                return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual UI to edit action segmentation boundaries and statuses video by video."
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
            "Root output directory. The tool loads auto segments from <output-dir>/segments, "
            "loads manual overrides from <output-dir>/manual_segments, and writes edits back to manual_segments."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_paths = collect_video_paths(args)
    for index, video_path in enumerate(video_paths, start=1):
        print(f"[{index}/{len(video_paths)}] Opening manual segment UI for {video_path.name}")
        editor = ManualSegmentEditor(video_path, args.output_dir, video_index=index, video_total=len(video_paths))
        result = editor.run()
        if result == "next":
            continue
        if result == "save_quit":
            print("Saved current video and quit.")
            break
        if result == "discard_quit":
            print("Quit without saving current video.")
            break


if __name__ == "__main__":
    main()
