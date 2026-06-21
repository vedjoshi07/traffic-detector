import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._ensure_header()

    def _ensure_header(self):
        # Create file with header if it doesn't exist
        if not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "zone_id", "zone_name", "vehicle_count", "density_level", "track_ids"])

    def log(self, zone_id: str, zone_name: str, count: int, level: str, track_ids: list):
        timestamp = datetime.now().isoformat(timespec='seconds')
        track_ids_str = ",".join(map(str, track_ids))
        
        # Append immediately (file writing is fast enough for 1 row, but in a heavy pipeline we could buffer)
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, zone_id, zone_name, count, level, track_ids_str])
