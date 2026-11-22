import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Danh sách các cặp điểm cần nối với nhau để tạo thành khung xương bàn tay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (13, 17), (17, 18), (18, 19), (19, 20),# Ngón út
    (0, 17)                                # Cạnh lòng bàn tay
]

class HandTracker:
    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        num_hands=2,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        # 1. Trỏ đường dẫn tới file model .task
        base_options = python.BaseOptions(model_asset_path=model_path)

        # 2. Cấu hình chế độ chạy VIDEO để AI bám theo tay mượt hơn
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # 3. Khởi tạo bộ nhận diện
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame):
        """Nhận vào 1 frame BGR từ OpenCV, trả về kết quả HandLandmarkerResult."""
        # Chuyển hệ màu từ BGR (OpenCV) sang RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Tạo mốc thời gian (ms) luôn tăng dần cho chế độ VIDEO
        current_ms = int(time.time() * 1000)
        timestamp_ms = max(self._last_timestamp_ms + 1, current_ms)
        self._last_timestamp_ms = timestamp_ms

        # Chạy AI nhận diện
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return result

    def get_pixel_landmarks(self, result, frame_shape):
        """
        Chuyển đổi tọa độ chuẩn hóa (0.0 - 1.0) sang tọa độ pixel thực (x, y).
        Trả về danh sách các bàn tay, mỗi bàn tay gồm 21 điểm (id, cx, cy).
        """
        h, w, _ = frame_shape
        all_hands = []

        if not result.hand_landmarks:
            return all_hands

        for hand_landmarks in result.hand_landmarks:
            hand_points = []
            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append((idx, cx, cy))
            all_hands.append(hand_points)

        return all_hands

    def draw_landmarks(self, frame, result):
        """Vẽ khung xương và 21 điểm khớp lên khung hình."""
        all_hands = self.get_pixel_landmarks(result, frame.shape)

        for hand_points in all_hands:
            # 1. Vẽ các đường nối xương trước (màu trắng)
            for start_idx, end_idx in HAND_CONNECTIONS:
                _, x1, y1 = hand_points[start_idx]
                _, x2, y2 = hand_points[end_idx]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # 2. Vẽ 21 chấm tròn lên trên (màu xanh lá)
            for idx, cx, cy in hand_points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        return frame

    def close(self):
        self.detector.close()