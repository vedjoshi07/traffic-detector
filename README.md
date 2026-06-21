# Traffic Density Estimator

A real-time, Streamlit-based application for estimating traffic density per lane using YOLOv8 and ByteTrack.

## Architecture & Features
- **YOLOv8** for vehicle detection
- **ByteTrack** (via `supervision`) for robust vehicle tracking across frames to ensure stable counting
- **Polygon Zones** for per-lane isolated analysis
- **Hysteresis Logic** to prevent density level flickering between states (e.g. LOW/MEDIUM bounds)
- **Live Heatmap** overlay to track congestion
- **Threaded Pipeline**: A producer thread handles the heavy CV workload while the Streamlit UI polls it, maintaining a responsive dashboard.
- **Analytics Dashboard**: View historical peak hours and cumulative heatmaps.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Assets**:
   - Place a test video at `data/sample.mp4`.
   - Place an alert sound at `assets/alert.mp3`.
   - Place your YOLO models in `models/` (e.g., `yolov8n.pt`, `yolov8s.pt`).

3. **Configure Zones**:
   Run the zone editor to draw lane polygons on your camera view:
   ```bash
   python -m tools.zone_editor
   ```
   - Left click to draw points.
   - Press `n` to finish a zone and enter its name.
   - Press `s` to save to `data/zones.json`.

4. **Run Application**:
   ```bash
   streamlit run app.py
   ```

## Configuration (`config.yaml`)
You can tweak model confidence, hysteresis frames, default threshold limits, and log file paths directly in `config.yaml`. The Streamlit sidebar allows on-the-fly modifications to confidence, processing speed, and alert levels.
