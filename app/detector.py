from ultralytics import YOLO
import cv2
import os

class FaceDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.5):
        os.makedirs("models", exist_ok=True)

        if not os.path.exists(model_path):
            print("[Detector] Downloading yolov8n.pt via ultralytics ...")
            self.model = YOLO("yolov8n.pt")   # auto-downloads from ultralytics servers
            # save to models/ folder for next time
            import shutil
            src = os.path.join(os.getcwd(), "yolov8n.pt")
            if os.path.exists(src):
                shutil.move(src, model_path)
                print(f"[Detector] Saved to {model_path}")
        else:
            self.model = YOLO(model_path)

        self.conf = conf

    def detect(self, frame):
        """
        Returns list of dicts: {bbox: [x1,y1,x2,y2], conf: float}
        Uses class 0 = person, detects face region from upper body.
        """
        results = self.model(frame, verbose=False)[0]
        faces   = []
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # crop only upper 1/3 of person bbox = face region
                face_h = (y2 - y1) // 3
                faces.append({
                    "bbox": [x1, y1, x2, y1 + face_h],
                    "conf": float(box.conf[0])
                })
        return faces

    def crop_face(self, frame, bbox, padding=10):
        """Crop face with optional padding, clamped to frame bounds."""
        h, w = frame.shape[:2]
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(w, bbox[2] + padding)
        y2 = min(h, bbox[3] + padding)
        return frame[y1:y2, x1:x2]