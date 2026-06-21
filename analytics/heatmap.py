import numpy as np
import cv2
import supervision as sv
import pandas as pd
import plotly.express as px

class LiveHeatmap:
    def __init__(self, resolution: tuple, decay: float = 0.98):
        self.width, self.height = resolution
        self.decay = decay
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, detections: sv.Detections):
        # Decay existing
        self.accumulator *= self.decay
        
        if len(detections) > 0:
            # Add Gaussian blobs at centroids
            centroids = detections.get_anchors_coordinates(sv.Position.CENTER)
            for x, y in centroids:
                x, y = int(x), int(y)
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Simple 2D Gaussian approximation
                    cv2.circle(self.accumulator, (x, y), 20, 1.0, -1)
                    
        # Apply slight blur to smooth out the blobs
        self.accumulator = cv2.GaussianBlur(self.accumulator, (15, 15), 0)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if np.max(self.accumulator) > 0:
            norm_heatmap = cv2.normalize(self.accumulator, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
            
            # Create a mask where accumulator is > low threshold to avoid tinting the whole image
            mask = norm_heatmap > 10
            
            # Blend
            result = frame.copy()
            alpha = 0.5
            result[mask] = cv2.addWeighted(result, 1 - alpha, color_heatmap, alpha, 0)[mask]
            return result
        return frame

def generate_peak_hours_chart(log_path: str):
    try:
        df = pd.read_csv(log_path, parse_dates=["timestamp"])
        if df.empty:
            return None
            
        df["hour"] = df["timestamp"].dt.hour
        
        # Group by hour and zone, calc mean
        agg = df.groupby(["hour", "zone_name"])["vehicle_count"].mean().reset_index()
        
        fig = px.bar(
            agg, 
            x="hour", 
            y="vehicle_count", 
            color="zone_name",
            barmode="group",
            title="Average Vehicle Density by Hour",
            labels={"hour": "Hour of Day", "vehicle_count": "Avg Vehicles"}
        )
        return fig
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def generate_session_heatmap_chart(log_path: str):
    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return None
            
        # Group by zone, calc overall max density
        agg = df.groupby("zone_name")["vehicle_count"].max().reset_index()
        
        fig = px.bar(
            agg,
            x="zone_name",
            y="vehicle_count",
            color="vehicle_count",
            color_continuous_scale="Reds",
            title="Max Vehicle Congestion Per Zone"
        )
        return fig
    except Exception as e:
        print(f"Error generating heatmap chart: {e}")
        return None
