# utils/detect.py
from ultralytics import YOLO
import cv2

def detect_players(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        players = [box for box in results.boxes.data if int(box[-1]) == 0]  # Assuming class 0 = player

        for box in players:
            x1, y1, x2, y2, conf, cls = box[:6]
            detections.append({
                "frame": frame_idx,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf)
            })

        frame_idx += 1
    cap.release()
    return detections
