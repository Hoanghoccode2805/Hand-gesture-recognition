import cv2
import joblib
import numpy as np
from pathlib import Path
from src.preprocessing import Preprocessor
from src.hand_tracking import HandTracker

BASE_DIR = Path(__file__).resolve().parent
TASK_MODEL_PATH = str(BASE_DIR / "models" / "hand_landmarker.task")
GESTURE_MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

CLASS_NAMES = {
    0: "Open Palm",
    1: "Fist",
    2: "Peace",
    3: "Pointing",
}

CONFIDENCE_THRESHOLD = 0.70

def extract_raw_features(result):
    if not result.hand_landmarks:
        return None

    hand_landmarks = result.hand_landmarks[0]
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features

def main():
    if not GESTURE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {GESTURE_MODEL_PATH}. "
            "Please run src/train_model.py to save the model first."
        )
    model = joblib.load(GESTURE_MODEL_PATH)


    cap = cv2.VideoCapture(0)
    tracker = HandTracker(
        model_path="models/hand_landmarker.task",
        num_hands=2
    )

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    try: 
        window_name = "Hand Gesture Data Collector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 500)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            result = tracker.detect(frame)
            frame = tracker.draw_landmarks(frame, result)
            
            display_text = "No Hand"
            text_color = (200, 200, 200)

            
            raw_features = extract_raw_features(result)
            if raw_features is not None:
                norm_features = Preprocessor.normalize_landmarks(raw_features)
                probs = model.predict_proba([norm_features])[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    gesture_name = CLASS_NAMES.get(pred_idx, "Unknown")
                    display_text = f"{gesture_name} ({confidence * 100:.1f}%)"
                    text_color = (0, 255, 0)
                else:
                    display_text = f"Uncertain ({confidence * 100:.1f}%)"
                    text_color = (0, 255, 255)

            cv2.putText(frame, f"Gesture: {display_text}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            cv2.putText(
                    frame,
                    "Press [Q] to quit",
                    (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

if __name__ == "__main__":
    main()