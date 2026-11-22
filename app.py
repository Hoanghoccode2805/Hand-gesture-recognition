import cv2
from src.hand_tracking import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(
        model_path="models/hand_landmarker.task",
        num_hands=2
    )

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)

        result = tracker.detect(frame)
        hands = tracker.get_pixel_landmarks(result, frame.shape)
        if hands:
            # Ví dụ: Lấy tọa độ đầu ngón trỏ (điểm số 8) của bàn tay đầu tiên
            _, index_x, index_y = hands[0][8]
            cv2.putText(
                frame, f"Index Tip: ({index_x}, {index_y})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        frame = tracker.draw_landmarks(frame, result)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()