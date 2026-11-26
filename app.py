from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from src.air_drawing import AirCanvas
from src.hand_tracking import HandTracker
from src.preprocessing import Preprocessor
from src.train_mlp import GestureMLP


BASE_DIR = Path(__file__).resolve().parent
TASK_MODEL_PATH = str(BASE_DIR / "models" / "hand_landmarker.task")

# use model mlp
MODEL_TYPE = "mlp"
RF_MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
MLP_MODEL_PATH = BASE_DIR / "models" / "mlp.pth"

CLASS_NAMES = {
    0: "Open Palm",
    1: "Fist",
    2: "Peace",
    3: "Pointing",
}

CONFIDENCE_THRESHOLD = 0.75

# HELPER FUNCTIONS FOR MODEL LOADING & PREDICTION
def load_gesture_model(model_type):
    if model_type == "rf":
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(f"Not found model at {RF_MODEL_PATH}")
        return joblib.load(RF_MODEL_PATH), None

    if model_type == "mlp":
        if not MLP_MODEL_PATH.exists():
            raise FileNotFoundError(f"Not found model at {MLP_MODEL_PATH}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GestureMLP(input_size=63, num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(
            torch.load(MLP_MODEL_PATH, map_location=device, weights_only=True)
        )
        model.eval()
        return model, device

    raise ValueError("MODEL_TYPE only accepts 'mlp' or 'rf'")


def predict_proba(model, model_type, device, norm_features):
    if model_type == "rf":
        return model.predict_proba([norm_features])[0]

    input_tensor = (
        torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        logits = model(input_tensor)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


def extract_raw_features(result):
    if not result.hand_landmarks:
        return None
    hand_landmarks = result.hand_landmarks[0]
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def main():
    model, device = load_gesture_model(MODEL_TYPE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    tracker = HandTracker(model_path=TASK_MODEL_PATH, num_hands=1)

    # Initialize the drawing surface and anti-flicker filter BEFORE the loop
    canvas = AirCanvas(brush_thickness=6)
    pred_history = deque(maxlen=5)

    try:
        window_name = f"Hand Gesture Air Drawing ({MODEL_TYPE.upper()})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 500)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = tracker.detect(frame)
            frame = tracker.draw_landmarks(frame, result)

            # Lấy tọa độ pixel đầu ngón trỏ (khớp số 8)
            index_tip = None
            if result.hand_landmarks:
                lm8 = result.hand_landmarks[0][8]
                h, w, _ = frame.shape
                index_tip = (int(lm8.x * w), int(lm8.y * h))

            display_text = "No Hand"
            text_color = (200, 200, 200)
            active_gesture = None

            raw_features = extract_raw_features(result)
            if raw_features is not None:
                norm_features = Preprocessor.normalize_landmarks(raw_features)
                probs = predict_proba(model, MODEL_TYPE, device, norm_features)
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    pred_history.append(pred_idx)
                    stable_idx = Counter(pred_history).most_common(1)[0][0]
                    active_gesture = CLASS_NAMES.get(stable_idx)
                    display_text = f"{active_gesture} ({confidence * 100:.1f}%)"
                    text_color = (0, 255, 0)
                else:
                    pred_history.clear()
                    display_text = f"Uncertain ({confidence * 100:.1f}%)"
                    text_color = (0, 255, 255)
            else:
                pred_history.clear()

            # Cập nhật nét vẽ và đè canvas lên frame
            action_msg = canvas.update(frame.shape, active_gesture, index_tip)
            frame = canvas.overlay(
                frame,
                index_tip=index_tip,
                is_drawing=(active_gesture == "Pointing"),
            )

            # Hiển thị thông tin HUD
            cv2.putText(
                frame,
                f"Gesture: {display_text}",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                text_color,
                2,
            )
            cv2.putText(
                frame,
                f"Color: {canvas.current_color_name} | {action_msg}",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                canvas.current_color_bgr,
                2,
            )
            cv2.putText(
                frame,
                "[Pointing]: Draw | [Fist]: Pause | [Peace]: Color | [Open Palm/C]: Clear | [Q]: Quit",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                canvas.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()