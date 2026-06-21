import supervision as sv

class Tracker:
    def __init__(self):
        # Initialize ByteTrack
        self.tracker = sv.ByteTrack()

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        # Update tracker with current detections
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections
