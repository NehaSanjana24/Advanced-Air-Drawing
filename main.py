from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np


WINDOW_NAME = "Advanced Air Drawing"
CAMERA_INDEX = 0
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

PALETTE = [
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
    ("Orange", (0, 165, 255)),
]

MIN_THICKNESS = 4
MAX_THICKNESS = 34
BRUSH_THICKNESS_DEFAULT = 10
ERASER_THICKNESS_DEFAULT = 28

PALETTE_AREA = (20, 20, 320, 76)
THICKNESS_AREA = (610, 110, 660, 420)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


@dataclass(frozen=True)
class Point:
    x: int
    y: int


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def distance(first: Point, second: Point) -> float:
    return float(np.hypot(first.x - second.x, first.y - second.y))


def midpoint(first: Point, second: Point) -> Point:
    return Point((first.x + second.x) // 2, (first.y + second.y) // 2)


def inside_rect(point: Point, rect: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    return left <= point.x <= right and top <= point.y <= bottom


def map_range(value: float, input_min: float, input_max: float, output_min: float, output_max: float) -> int:
    if input_max == input_min:
        return int(output_min)
    normalized = (value - input_min) / (input_max - input_min)
    normalized = max(0.0, min(1.0, normalized))
    return int(output_min + normalized * (output_max - output_min))


def finger_states(landmarks, handedness_label: str) -> list[bool]:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if handedness_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    fingers_up = [
        thumb_up,
        landmarks[8].y < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y < landmarks[14].y,
        landmarks[20].y < landmarks[18].y,
    ]
    return fingers_up


def ensure_model_downloaded() -> Path:
    if not MODEL_PATH.exists():
        print("Downloading MediaPipe hand landmarker model...")
        urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def handedness_label_from(result, index: int = 0) -> str:
    if not result.handedness or not result.handedness[index]:
        return "Right"

    handedness = result.handedness[index][0]
    for attribute in ("category_name", "categoryName", "display_name", "displayName"):
        value = getattr(handedness, attribute, None)
        if value:
            return value
    return "Right"


class AirDrawingApp:
    def __init__(self) -> None:
        self.canvas: np.ndarray | None = None
        self.selected_color = PALETTE[1][1]
        self.brush_thickness = BRUSH_THICKNESS_DEFAULT
        self.eraser_thickness = ERASER_THICKNESS_DEFAULT
        self.previous_point: Point | None = None
        self.status_text = "Draw with index finger, pinch to select, open palm to clear"
        self.status_until = 0.0
        self.clear_cooldown_until = 0.0

        model_path = ensure_model_downloaded()
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )

    def set_status(self, text: str, duration: float = 1.4) -> None:
        self.status_text = text
        self.status_until = monotonic() + duration

    def clear_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas.fill(0)
        self.previous_point = None
        self.set_status("Canvas cleared")

    def draw_hand_skeleton(self, frame: np.ndarray, landmarks) -> None:
        height, width = frame.shape[:2]
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]

        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start_point = points[connection.start]
            end_point = points[connection.end]
            cv2.line(frame, start_point, end_point, (80, 220, 255), 2, cv2.LINE_AA)

        for x_coord, y_coord in points:
            cv2.circle(frame, (x_coord, y_coord), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x_coord, y_coord), 5, (20, 20, 20), 1, cv2.LINE_AA)

    def draw_toolbar(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        height, width = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (width, 110), (18, 18, 24), -1)
        cv2.rectangle(overlay, (10, 10), (width - 10, 100), (30, 32, 40), 2)

        left, top, right, bottom = PALETTE_AREA
        cv2.putText(overlay, "Colors", (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)

        button_width = 44
        button_gap = 10
        for index, (name, color) in enumerate(PALETTE):
            x1 = left + index * (button_width + button_gap)
            y1 = top + 12
            x2 = x1 + button_width
            y2 = y1 + button_width
            border_color = (255, 255, 255) if color == self.selected_color else (70, 70, 80)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 2)
            cv2.putText(overlay, str(index + 1), (x1 + 14, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

        t_left, t_top, t_right, t_bottom = THICKNESS_AREA
        cv2.putText(overlay, "Thickness", (t_left - 10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(overlay, (t_left + 20, t_top), (t_left + 30, t_bottom), (90, 90, 100), -1)
        marker_y = map_range(self.brush_thickness, MIN_THICKNESS, MAX_THICKNESS, t_bottom, t_top)
        cv2.circle(overlay, (t_left + 25, marker_y), 10, (255, 255, 255), -1)
        cv2.circle(overlay, (t_left + 25, marker_y), 10, (20, 20, 20), 2)

        cv2.putText(overlay, f"Brush {self.brush_thickness}px", (t_left - 8, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(overlay, "Pinch a color or drag the brush handle", (390, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(overlay, "Index finger draws | open palm clears | Q or ESC quits", (390, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if self.status_until > monotonic():
            cv2.putText(frame, self.status_text, (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def select_palette_color(self, point: Point) -> bool:
        left, top, _, _ = PALETTE_AREA
        button_width = 44
        button_gap = 10

        for index, (_, color) in enumerate(PALETTE):
            x1 = left + index * (button_width + button_gap)
            y1 = top + 12
            x2 = x1 + button_width
            y2 = y1 + button_width
            if x1 <= point.x <= x2 and y1 <= point.y <= y2:
                self.selected_color = color
                self.set_status(f"Selected {PALETTE[index][0]}")
                return True
        return False

    def adjust_thickness(self, point: Point) -> bool:
        left, top, _, bottom = THICKNESS_AREA
        rail_left = left + 15
        rail_right = left + 35
        if rail_left <= point.x <= rail_right and top <= point.y <= bottom:
            self.brush_thickness = map_range(point.y, bottom, top, MIN_THICKNESS, MAX_THICKNESS)
            self.set_status(f"Brush thickness {self.brush_thickness}px")
            return True
        return False

    def render_hand_landmarks(self, frame: np.ndarray, results) -> None:
        if not results.hand_landmarks:
            self.previous_point = None
            return

        height, width = frame.shape[:2]
        hand_landmarks = results.hand_landmarks[0]
        handedness = handedness_label_from(results)

        self.draw_hand_skeleton(frame, hand_landmarks)

        states = finger_states(hand_landmarks, handedness)
        landmarks = hand_landmarks
        index_point = Point(int(landmarks[8].x * width), int(landmarks[8].y * height))
        thumb_point = Point(int(landmarks[4].x * width), int(landmarks[4].y * height))
        pinch_point = midpoint(index_point, thumb_point)
        pinch_active = distance(index_point, thumb_point) < max(32, width * 0.05)

        if pinch_active:
            if self.select_palette_color(pinch_point):
                self.previous_point = None
                return
            if self.adjust_thickness(pinch_point):
                self.previous_point = None
                return

        index_up = states[1]
        middle_up = states[2]
        ring_up = states[3]
        pinky_up = states[4]

        open_palm = index_up and middle_up and ring_up and pinky_up
        erase_mode = index_up and middle_up and not ring_up and not pinky_up
        draw_mode = index_up and not middle_up and not ring_up and not pinky_up

        if open_palm and monotonic() >= self.clear_cooldown_until:
            self.clear_canvas()
            self.clear_cooldown_until = monotonic() + 1.0
            return

        if self.canvas is None:
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if draw_mode:
            if self.previous_point is not None:
                cv2.line(self.canvas, (self.previous_point.x, self.previous_point.y), (index_point.x, index_point.y), self.selected_color, self.brush_thickness)
            self.previous_point = index_point
            return

        if erase_mode:
            if self.previous_point is not None:
                cv2.line(self.canvas, (self.previous_point.x, self.previous_point.y), (index_point.x, index_point.y), (0, 0, 0), self.eraser_thickness)
            self.previous_point = index_point
            self.set_status("Eraser active")
            return

        self.previous_point = None

    def compose_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.canvas is not None:
            frame = cv2.addWeighted(frame, 0.72, self.canvas, 1.0, 0)
        self.draw_toolbar(frame)
        return frame

    def run(self) -> None:
        capture = cv2.VideoCapture(CAMERA_INDEX)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not capture.isOpened():
            raise RuntimeError("Could not open the webcam. Make sure another app is not using it.")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)

                if self.canvas is None:
                    self.canvas = np.zeros_like(frame)
                elif self.canvas.shape != frame.shape:
                    self.canvas = np.zeros_like(frame)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(monotonic() * 1000)
                results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                self.render_hand_landmarks(frame, results)
                view = self.compose_frame(frame)

                cv2.imshow(WINDOW_NAME, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c"):
                    self.clear_canvas()
                if key == ord("+") or key == ord("="):
                    self.brush_thickness = clamp(self.brush_thickness + 2, MIN_THICKNESS, MAX_THICKNESS)
                    self.set_status(f"Brush thickness {self.brush_thickness}px")
                if key == ord("-") or key == ord("_"):
                    self.brush_thickness = clamp(self.brush_thickness - 2, MIN_THICKNESS, MAX_THICKNESS)
                    self.set_status(f"Brush thickness {self.brush_thickness}px")
        finally:
            capture.release()
            cv2.destroyAllWindows()


def main() -> None:
    app = AirDrawingApp()
    app.run()


if __name__ == "__main__":
    main()