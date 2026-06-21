import cv2
import json
import yaml
import os
import numpy as np

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_editor():
    config = load_config()
    video_path = config["video"]["default_source"]
    zones_path = config["zones_path"]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        return
    cap.release()

    frame_height, frame_width = frame.shape[:2]
    zones = []
    current_polygon = []
    
    window_name = "Zone Editor"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append([x, y])

    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Zone Editor ---")
    print("Left click to add points to the current zone polygon.")
    print("Press 'n' to finish current zone and start a new one.")
    print("Press 's' to save all zones and exit.")
    print("Press 'q' to quit without saving.")

    while True:
        display_frame = frame.copy()
        
        # Draw saved zones
        for i, zone in enumerate(zones):
            pts = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(display_frame, zone["name"], tuple(zone["polygon"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw current polygon
        if len(current_polygon) > 0:
            pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], False, (0, 0, 255), 2)
            for pt in current_polygon:
                cv2.circle(display_frame, tuple(pt), 4, (0, 0, 255), -1)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            if len(current_polygon) >= 3:
                zone_name = input(f"Enter name for Zone {len(zones) + 1} (e.g., Lane Left): ")
                zone_id = zone_name.lower().replace(" ", "_")
                zones.append({
                    "id": zone_id,
                    "name": zone_name,
                    "polygon": current_polygon.copy()
                })
                current_polygon.clear()
                print(f"Zone '{zone_name}' added.")
            else:
                print("A zone needs at least 3 points.")
        elif key == ord('s'):
            zone_data = {
                "zones": zones,
                "frame_width": frame_width,
                "frame_height": frame_height
            }
            with open(zones_path, "w") as f:
                json.dump(zone_data, f, indent=2)
            print(f"Saved {len(zones)} zones to {zones_path}")
            break
        elif key == ord('q'):
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_editor()
