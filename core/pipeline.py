import threading
import queue
import time
import cv2
import numpy as np
import supervision as sv

from core.detector import Detector
from core.tracker import Tracker
from core.zones import ZoneManager
from core.density import DensityClassifier
from core.alerts import AlertManager
from core.csv_logger import CSVLogger
from analytics.heatmap import LiveHeatmap

class Pipeline(threading.Thread):
    def __init__(self, config: dict, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self.config = config
        self.out_queue = out_queue
        self.stop_event = stop_event

        self.video_source = self.config["video"]["default_source"]
        
        # Try to convert to int if it's a camera index
        try:
            self.video_source = int(self.video_source)
        except ValueError:
            pass

        self.process_every_n = self.config["video"].get("process_every_n_frames", 1)
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                self._push_error(f"Could not open video source: {self.video_source}")
                return

            # Initialize components inside the thread (except heavy ones passed in, but we init them here for safety)
            detector = Detector(
                self.config["model"]["weights"], 
                self.config["model"]["confidence"], 
                self.config["model"]["classes"],
                self.config["model"]["device"]
            )
            tracker = Tracker()
            
            # Setup Zones
            zone_manager = ZoneManager(self.config["zones_path"])
            
            # Read first frame to get resolution
            ret, frame = cap.read()
            if not ret:
                self._push_error("Failed to read from video source.")
                cap.release()
                return
                
            frame_resolution = (frame.shape[1], frame.shape[0])
            zones_dict = zone_manager.build_polygon_zones(frame_resolution)
            
            # FPS throttling
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            frame_delay = 1.0 / fps
            
            # Annotators
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            zone_annotators = {
                zid: sv.PolygonZoneAnnotator(zone=z["zone"], color=sv.Color.WHITE)
                for zid, z in zones_dict.items()
            }
            
            # Density & Alerts
            classifier = DensityClassifier(self.config["density_thresholds"], self.config["hysteresis_frames"])
            alert_manager = AlertManager(self.config.get("alert_on_level", "HIGH"))
            csv_logger = CSVLogger(self.config["log_path"])
            
            # Heatmap
            live_heatmap = LiveHeatmap(frame_resolution)

            frame_count = 0
            last_detections = sv.Detections.empty()

            # Process loop
            # Rewind to start since we read 1 frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while not self.stop_event.is_set():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                    
                frame_count += 1
                
                if frame_count % self.process_every_n == 0:
                    detections = detector.detect(frame)
                    last_detections = tracker.update_with_detections(detections)

                annotated_frame = frame.copy()
                
                # Update Heatmap
                live_heatmap.update(last_detections)
                annotated_frame = live_heatmap.draw(annotated_frame)
                
                # Annotate Boxes
                labels = [
                    f"#{tracker_id} {detector.model.names[class_id]}"
                    for class_id, tracker_id in zip(last_detections.class_id, last_detections.tracker_id)
                ] if last_detections.tracker_id is not None else []
                
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=last_detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=last_detections, labels=labels)

                # Process Zones
                zone_states = {}
                for zone_id, z_data in zones_dict.items():
                    zone = z_data["zone"]
                    config = z_data["config"]
                    
                    # Trigger zone
                    mask = zone.trigger(detections=last_detections)
                    
                    # Track IDs in zone
                    if last_detections.tracker_id is not None:
                        in_zone_track_ids = last_detections.tracker_id[mask].tolist()
                    else:
                        in_zone_track_ids = []
                        
                    count = len(in_zone_track_ids)
                    
                    # Classify
                    level = classifier.classify(zone_id, count)
                    
                    # Alerts
                    alert_manager.check(zone_id, level)
                    
                    # Log
                    if frame_count % 30 == 0 or alert_manager.just_triggered: 
                        # Log periodically (e.g. 1 per sec if 30fps) or if alert triggered
                        csv_logger.log(zone_id, config.name, count, level, in_zone_track_ids)
                    
                    # Annotate zone
                    annotated_frame = zone_annotators[zone_id].annotate(scene=annotated_frame)
                    
                    # Add to state
                    zone_states[zone_id] = {
                        "name": config.name,
                        "count": count,
                        "level": level
                    }

                # Get just triggered alerts
                triggered_alerts = alert_manager.get_and_clear_just_triggered()
                active_alerts = alert_manager.get_active_alerts()

                # Compress to JPEG to save Streamlit/WebSocket bandwidth
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Push to queue (drop if full to avoid lag)
                state = {
                    "frame_jpg": buffer.tobytes(),
                    "zone_states": zone_states,
                    "triggered_alerts": triggered_alerts,
                    "active_alerts": active_alerts
                }
                
                if self.out_queue.full():
                    try:
                        self.out_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.out_queue.put(state)
                
                # Throttle to roughly 1x real-time speed
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
            cap.release()
            self.out_queue.put({"status": "stopped"})

        except Exception as e:
            self._push_error(str(e))
            
    def _push_error(self, message: str):
        if self.out_queue.full():
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
        self.out_queue.put({"error": message})
