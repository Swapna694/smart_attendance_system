import cv2
import os
import sys
import numpy as np
sys.path.insert(0, ".")

from app.recognizer import FaceRecognizer
from app.detector   import FaceDetector
from app.database   import save_person_embedding, test_connection

captured    = 0
should_capture = False

def on_mouse(event, x, y, flags, param):
    global should_capture
    if event == cv2.EVENT_LBUTTONDOWN:
        should_capture = True

def register(name: str, employee_id: str, department: str,
             image_folder: str = None, use_camera: bool = False):
    global captured, should_capture

    recognizer = FaceRecognizer()
    detector   = FaceDetector()
    embeddings = []

    if image_folder:
        for fname in os.listdir(image_folder):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(image_folder, fname))
            if img is None:
                continue
            faces = detector.detect(img)
            if faces:
                crop = detector.crop_face(img, faces[0]["bbox"])
                emb  = recognizer.get_embedding(crop)
                if emb is not None:
                    embeddings.append(emb)

    elif use_camera:
        cap   = cv2.VideoCapture(0)
        captured = 0
        should_capture = False

        cv2.namedWindow("Register Face")
        cv2.setMouseCallback("Register Face", on_mouse)  # ← click to capture

        print("[Register] Press SPACE or LEFT CLICK on the window to capture.")

        while captured < 20:
            ret, frame = cap.read()
            if not ret:
                break

            # draw instruction text
            cv2.putText(frame, f"Captures: {captured}/20", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE or CLICK to capture", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Register Face", frame)

            key = cv2.waitKey(1) & 0xFF

            # trigger on SPACE key OR mouse click
            if key == ord(" ") or key == 32 or should_capture:
                should_capture = False
                faces = detector.detect(frame)
                if faces:
                    crop = detector.crop_face(frame, faces[0]["bbox"])
                    emb  = recognizer.get_embedding(crop)
                    if emb is not None:
                        embeddings.append(emb)
                        captured += 1
                        print(f"  Captured {captured}/20")
                        # flash green border to confirm capture
                        green = frame.copy()
                        cv2.rectangle(green, (0,0),
                                      (frame.shape[1], frame.shape[0]),
                                      (0,255,0), 15)
                        cv2.imshow("Register Face", green)
                        cv2.waitKey(200)
                    else:
                        print("  [!] No face embedding found, try again.")
                else:
                    print("  [!] No face detected in frame, try again.")

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    if not embeddings:
        print("[Register] No valid embeddings captured. Aborting.")
        return

    import numpy as np
    mean_emb   = np.mean(embeddings, axis=0)
    emb_bytes  = recognizer.embedding_to_bytes(mean_emb)
    save_person_embedding(name, employee_id, department, emb_bytes)
    print(f"[Register] ✅ '{name}' enrolled with {len(embeddings)} samples.")


if __name__ == "__main__":
    test_connection()
    register(
        name         = "Venkatramana",
        employee_id  = "EMP003",
        department   = "Engineering",
        use_camera   = True
    )