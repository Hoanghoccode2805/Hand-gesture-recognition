import cv2
import pandas as pd
import time
import numpy as np

class AirCanvas:
    def __init__(self, brush_thickness=6):
        self.canvas = None
        self.prev_point = None
        self.brush_thickness = brush_thickness

        self.colors = [
            ("Blue", (255, 100, 0)),
            ("Green", (0, 220, 0)),
            ("Red", (0, 0, 255)),
            ("Yellow", (0, 230, 255)),
            ("White", (255, 255, 255)),
        ]
        self.color_idx = 0
        self.last_color_change = 0.0
        self.color_cooldown = 1.0           # 1 giây mới cho đổi màu tiếp

        self.clear_start_time = None
        self.clear_hold_duration = 0.8      # Giữ Open Palm 0.8 giây để xóa

    @property
    def current_color_name(self):
        return self.colors[self.color_idx][0]

    @property
    def current_color_bgr(self):
        return self.colors[self.color_idx][1]

    def _ensure_canvas(self, frame_shape):
        """Automatically initialize a black-background canvas that matches the webcam dimensions on the first frame."""
        if self.canvas is None or self.canvas.shape != frame_shape:
            self.canvas = np.zeros(frame_shape, dtype=np.uint8)

    def clear(self):
        if self.canvas is not None:
            self.canvas[:] = 0
        self.prev_point = None

    def update(self, frame_shape, gesture_name, index_tip):
        """
        Update the drawing state based on the current gesture. 
        - gesture_name: 'Pointing', 'Fist', 'Peace', 'Open Palm', or None
        - index_tip: pixel coordinates (x, y) of the index finger tip (landmark 8), or None
        """
        self._ensure_canvas(frame_shape)
        now = time.time()
        status_msg = ""

        # 1. CLEAR BOARD gesture handling (Open Palm)
        if gesture_name == "Open Palm":
            self.prev_point = None
            if self.clear_start_time is None:
                self.clear_start_time = now
            elapsed = now - self.clear_start_time
            if elapsed >= self.clear_hold_duration:
                self.clear()
                self.clear_start_time = None
                status_msg = "Canvas Cleared!"
            else:
                status_msg = f"Hold to Clear ({elapsed:.1f}/{self.clear_hold_duration}s)"
            return status_msg
        else:
            self.clear_start_time = None        # Nếu không còn giơ Open Palm thì reset bộ đếm xóa

        # 2. COLOR-CHANGING gesture handling (Peace)
        if gesture_name == "Peace":
            self.prev_point = None
            if now - self.last_color_change >= self.color_cooldown:
                self.color_idx = (self.color_idx + 1) % len(self.colors)
                self.last_color_change = now
                status_msg = f"Color -> {self.current_color_name}"
            return status_msg

        # 3. DRAW gesture handling (Pointing)
        if gesture_name == "Pointing" and index_tip is not None:
            if self.prev_point is not None:
                cv2.line(
                    self.canvas,
                    self.prev_point,
                    index_tip,
                    self.current_color_bgr,
                    self.brush_thickness,
                    lineType=cv2.LINE_AA,  # Anti-aliasing for smoother strokes
                )
            self.prev_point = index_tip
            status_msg = "Drawing..."
            return status_msg

        # 4. Remaining cases (Fist, Uncertain, No Hand): Nhấc bút
        self.prev_point = None
        return status_msg

    def overlay(self, frame, index_tip=None, is_drawing=False):
        """Overlay the drawings from the canvas onto the webcam frame."""
        self._ensure_canvas(frame.shape)

        # Tạo mặt nạ (mask): chỗ nào trên canvas có màu (> 0) thì lấy màu của canvas,
        # chỗ nào màu đen (= 0) thì giữ nguyên ảnh webcam
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(bg, fg)

        # Draw a small circle at the tip of the index finger to serve as a stylus pointer.
        if index_tip is not None:
            radius = 8 if is_drawing else 5
            cv2.circle(combined, index_tip, radius, self.current_color_bgr, -1)
            cv2.circle(combined, index_tip, radius + 2, (255, 255, 255), 1)

        return combined


