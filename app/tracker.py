from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceTracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame):
        """
        detections: list of dicts from detector.detect()
        Returns confirmed tracks (DeepSort Track objects).
        """
        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            ds_input.append(([x1, y1, x2 - x1, y2 - y1], d["conf"], "face"))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)
        return [t for t in tracks if t.is_confirmed()]