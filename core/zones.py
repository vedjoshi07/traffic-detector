import json
import os
import supervision as sv
import numpy as np

class ZoneConfig:
    def __init__(self, zone_id: str, name: str, polygon: list):
        self.zone_id = zone_id
        self.name = name
        self.polygon = polygon

class ZoneManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.zones_data = []
        self.source_width = 1280
        self.source_height = 720
        self.load_zones()

    def load_zones(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: {self.config_path} not found. Returning empty zones.")
            return

        with open(self.config_path, "r") as f:
            data = json.load(f)
            
        self.source_width = data.get("frame_width", 1280)
        self.source_height = data.get("frame_height", 720)
        self.zones_data = [ZoneConfig(z["id"], z["name"], z["polygon"]) for z in data.get("zones", [])]

    def build_polygon_zones(self, frame_resolution: tuple) -> dict:
        """
        frame_resolution: (width, height)
        Returns: Dict mapping zone_id -> { 'config': ZoneConfig, 'zone': sv.PolygonZone }
        """
        target_width, target_height = frame_resolution
        scale_x = target_width / self.source_width if self.source_width else 1.0
        scale_y = target_height / self.source_height if self.source_height else 1.0

        built_zones = {}
        for zc in self.zones_data:
            # Scale polygon to target resolution
            scaled_poly = []
            for x, y in zc.polygon:
                scaled_poly.append([int(x * scale_x), int(y * scale_y)])
            
            scaled_poly_np = np.array(scaled_poly, np.int32)
            zone = sv.PolygonZone(polygon=scaled_poly_np)
            built_zones[zc.zone_id] = {
                "config": zc,
                "zone": zone
            }
            
        return built_zones
