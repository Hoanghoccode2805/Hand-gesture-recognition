import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        
    (0, 5), (5, 6), (6, 7), (7, 8),        
    (5, 9), (9, 10), (10, 11), (11, 12),   
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)                              
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
        # 1. Point to the path of .task model file
        base_options = python.BaseOptions(model_asset_path=model_path)

        # 2. Configure video playback mode
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,          # Mode VIDEO 
            num_hands=num_hands,                            # Number of hand need to detect 
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # 3. Initialize the recognizer
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0     

    def detect(self, frame):
        """Takes an BGR frame from OpenCV as input and returns a HandLandmarkerResult."""
        # Change color system from BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_ms = int(time.time() * 1000)        # Get current time in Laptop
        timestamp_ms = max(self._last_timestamp_ms + 1, current_ms)     
        self._last_timestamp_ms = timestamp_ms

        # Run AI to detect
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return result

    def get_pixel_landmarks(self, result, frame_shape):
        """
        Convert normalized coordinates (0.0 - 1.0) to actual pixel coordinates (x, y). 
        Returns a list of hands, with each hand consisting of 21 points (id, cx, cy)..
        """
        h, w, _ = frame_shape               # _ is a throwaway variable
        all_hands = []

        # When the hand is detected: result.hand_landmarks is a list of hands (each containing 21 landmarks)
        # When the hand is not detected: result.hand_landmarks is just []
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
        """Draw the skeleton and 21 joint points onto the frame."""
        all_hands = self.get_pixel_landmarks(result, frame.shape)

        for hand_points in all_hands:
            # 1. Draw the lines connecting the bones (in white)
            for start_idx, end_idx in HAND_CONNECTIONS:
                _, x1, y1 = hand_points[start_idx]
                _, x2, y2 = hand_points[end_idx]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # 2. Draw 21 small circles on top (in red)
            for idx, cx, cy in hand_points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        return frame

    def close(self):
        self.detector.close()