import time
import datetime
import cv2
import mediapipe as mp
import pathlib
from pathlib import Path
from hand_tracking import HandTracker
import csv
import pandas as pd

model_path = f"models/hand_landmarker.task"

class DataCollector:
    def __init__(
            self,
            csv_path = f"data/gesture_data.csv",
            save_inteval = 0.2,
    ):
        self.tracker = HandTracker(model_path= model_path, num_hands= 1)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_label = 0
        self.is_recording = False

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.class_names = {
            0 : "Open Palm",
            1 : "Fist",
            2 : "Peace",
            3 : "Pointing"
        }
        self.counts = 0 

        self.save_inteval = save_inteval
        self.last_save_time = 0.0
        self._init_csv()

    def _init_csv(self):
        columns = []
        path = Path(self.csv_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        is_empty_or_missing = not path.exists() or path.stat().st_size == 0

        if is_empty_or_missing:
            for i in range(0,21):
                columns.extend([f"x{i}", f"y{i}", f"z{i}" ])
            columns.extend(["label", "session_id"])

            with open(self.csv_path, mode="a",newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    def extract_features(self, result):
        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        hand_points = []
        for idx, lm in enumerate(hand_landmarks):
            hand_points.extend([lm.x, lm.y, lm.z])   

        return hand_points
    
    def save_sample(self, features):
        current_time = time.time()
        if current_time - self.last_save_time < self.save_inteval:
            return False

        if not features:
            return False
        
        data = features + [self.current_label, self.session_id]

        with open(self.csv_path, mode = "a", newline= "") as f:
            writer = csv.writer(f)
            writer.writerow(data)

        self.last_save_time = current_time
        self.counts += 1

        return True
    
    def handle_keypress(self, key):
        if key == ord('0'):
            self.current_label = 0
        elif key == ord('1'):
            self.current_label = 1
        elif key == ord('2'):
            self.current_label = 2
        elif key == ord('3'):
            self.current_label = 3
        elif key == ord(" "):
            self.is_recording = not self.is_recording
        elif key == ord("q"):
            return False
        
        return True

    def draw_ui(self, frame):
        class_name = self.class_names[self.current_label]
        status_text = "RECORDING" if self.is_recording else "PAUSED"
        status_color = (0, 0, 255) if self.is_recording else (0, 255, 255)

        cv2.putText(
            frame, f"Class: {self.current_label} - {class_name}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"Status: {status_text} | Saved: {self.counts}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )
        cv2.putText(
            frame, "[0-3]: Class | [SPACE]: Record/Pause | [Q]: Quit",
            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                result = self.tracker.detect(frame)

                frame = self.tracker.draw_landmarks(frame, result)

                if self.is_recording and result.hand_landmarks:
                    features = self.extract_features(result)
                    self.save_sample(features)

                frame = self.draw_ui(frame)
                cv2.imshow("Hand Gesture Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keypress(key):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
                    

