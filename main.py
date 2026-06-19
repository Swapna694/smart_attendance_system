import cv2
import sys
sys.path.insert(0, ".")

from app.detector        import FaceDetector
from app.recognizer      import FaceRecognizer
from app.tracker         import FaceTracker
from app.duplicate_check import DuplicateChecker
from app.database        import (
    test_connection, load_all_embeddings, log_attendance
)
from app.alerts import send_attendance_alert

def run():
    test_connection()

    detector   = FaceDetector("models/yolov8n.pt")
    recognizer = FaceRecognizer(threshold=0.9)
    tracker    = FaceTracker()
    dedup      = DuplicateChecker()

    rows = load_all_embeddings()
    recognizer.load_from_db(rows)

    identity_cache = {}

    cap = cv2.VideoCapture(0)
    print("[Main] Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── use MTCNN directly to find faces (more accurate than YOLO crop) ──
        from PIL import Image
        import torch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        boxes, _ = recognizer.mtcnn.detect(pil_img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # clamp to frame
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # ── identify ──────────────────────────────────────────────────
                pid, name = recognizer.identify(face_crop)

                # ── mark attendance ───────────────────────────────────────────
                if pid is not None and not dedup.already_marked(pid):
                    log_attendance(pid, name, 0)
                    dedup.mark(pid)
                    send_attendance_alert(name, department="Engineering")
                    print(f"[Attendance] Marked: {name} (ID {pid})")

                # ── draw box and name ─────────────────────────────────────────
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()