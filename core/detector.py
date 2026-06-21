import streamlit as st
from ultralytics import YOLO
import supervision as sv
import torch

@st.cache_resource
def load_yolo_model(weights_path: str, device: str = "auto") -> YOLO:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load model
    try:
        model = YOLO(weights_path)
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model from {weights_path}: {e}")
        raise e

class Detector:
    def __init__(self, weights_path: str, confidence: float, classes: list, device: str = "auto"):
        self.model = load_yolo_model(weights_path, device)
        self.confidence = confidence
        self.classes = classes

    def detect(self, frame) -> sv.Detections:
        # Run inference
        results = self.model(frame, verbose=False, conf=self.confidence, classes=self.classes)
        # Convert to supervision detections
        detections = sv.Detections.from_ultralytics(results[0])
        return detections
