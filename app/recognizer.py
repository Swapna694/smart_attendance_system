import torch
import numpy as np
import pickle
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

class FaceRecognizer:
    def __init__(self, threshold=0.9):
        self.mtcnn     = MTCNN(image_size=160, margin=20, keep_all=False)
        self.resnet    = InceptionResnetV1(pretrained="vggface2").eval()
        self.threshold = threshold
        self.db        = {}   # {name: (person_id, embedding_np)}

    # ── embedding ────────────────────────────────────────────────────────────

    def get_embedding(self, img_bgr):
        """Return 512-d numpy embedding or None if no face found."""
        img  = Image.fromarray(img_bgr[:, :, ::-1])   # BGR → RGB
        face = self.mtcnn(img)
        if face is None:
            return None
        with torch.no_grad():
            emb = self.resnet(face.unsqueeze(0))
        return emb[0].numpy()

    def embedding_to_bytes(self, emb_np):
        return pickle.dumps(emb_np)

    def bytes_to_embedding(self, raw_bytes):
        return pickle.loads(raw_bytes)

    # ── database load ────────────────────────────────────────────────────────

    def load_from_db(self, rows):
        """
        rows = [(id, name, embedding_bytes), ...]  from database.load_all_embeddings()
        """
        self.db = {}
        for person_id, name, emb_bytes in rows:
            emb = self.bytes_to_embedding(emb_bytes)
            self.db[name] = (person_id, emb)
        print(f"[Recognizer] Loaded {len(self.db)} known faces.")

    # ── identify ─────────────────────────────────────────────────────────────

    def identify(self, img_bgr):
        """
        Returns (person_id, name) or (None, 'Unknown').
        """
        query = self.get_embedding(img_bgr)
        if query is None:
            return None, "Unknown"

        best_name, best_pid, best_dist = "Unknown", None, self.threshold
        for name, (pid, emb) in self.db.items():
            dist = float(np.linalg.norm(query - emb))
            if dist < best_dist:
                best_name, best_pid, best_dist = name, pid, dist

        return best_pid, best_name