import streamlit as st
import yaml
import queue
import time
import os
import shutil
import plotly.graph_objects as go
import threading
from pathlib import Path

from core.pipeline import Pipeline
from analytics.heatmap import generate_peak_hours_chart, generate_session_heatmap_chart

# st.set_page_config MUST be first Streamlit call
st.set_page_config(page_title="Traffic Density Estimator", layout="wide", page_icon="🚦")

# ---- MODEL CHECK ----
@st.cache_resource
def ensure_default_model():
    """Verify yolov8n.pt is present in models/. If missing, download it."""
    import urllib.request
    model_path = Path("models/yolov8n.pt")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        try:
            urllib.request.urlretrieve(url, str(model_path))
        except Exception:
            # Fallback for macOS Python lacking SSL certificates
            import ssl
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx) as resp, open(str(model_path), "wb") as f:
                f.write(resp.read())
    return str(model_path)

ensure_default_model()

@st.cache_resource
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


# ---- SESSION STATE ----
if "running" not in st.session_state:
    st.session_state.running = False
if "pipeline_thread" not in st.session_state:
    st.session_state.pipeline_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "out_queue" not in st.session_state:
    st.session_state.out_queue = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None

# ---- SIDEBAR ----
st.sidebar.title("🚦 Controls")

# --- Video source ---
st.sidebar.markdown("### 🎬 Video Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload any traffic video. It will be saved locally and used as the source."
)

UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

if uploaded_file is not None:
    upload_path = UPLOADS_DIR / uploaded_file.name
    if not upload_path.exists() or upload_path.stat().st_size != uploaded_file.size:
        with open(str(upload_path), "wb") as f:
            f.write(uploaded_file.read())
    st.session_state.uploaded_video_path = str(upload_path)
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

# Determine default for the path input
default_source = st.session_state.uploaded_video_path or config["video"]["default_source"]
video_source = st.sidebar.text_input(
    "Or enter path / RTSP URL",
    value=default_source,
    help="File path, camera index (0), or RTSP stream URL. Overrides uploaded file if changed."
)

# If user manually changed the text input away from an uploaded file, clear the upload
if video_source != st.session_state.uploaded_video_path:
    st.session_state.uploaded_video_path = None

config["video"]["default_source"] = video_source
st.sidebar.markdown("---")

models = sorted([f for f in os.listdir("models") if f.endswith(".pt")]) if os.path.exists("models") else []
if not models:
    st.sidebar.warning("No .pt models found in models/. Add yolov8n.pt to proceed.")
    selected_model = None
else:
    default_idx = models.index("yolov8n.pt") if "yolov8n.pt" in models else 0
    selected_model = st.sidebar.selectbox("YOLOv8 Model", models, index=default_idx)

if selected_model:
    config["model"]["weights"] = os.path.join("models", selected_model)

confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=config["model"]["confidence"])
config["model"]["confidence"] = confidence

process_n = st.sidebar.slider("Process Every N Frames", min_value=1, max_value=10, value=config["video"].get("process_every_n_frames", 1))
config["video"]["process_every_n_frames"] = process_n

alert_level = st.sidebar.selectbox("Alert on Density", ["LOW", "MEDIUM", "HIGH"], index=["LOW", "MEDIUM", "HIGH"].index(config.get("alert_on_level", "HIGH")))
config["alert_on_level"] = alert_level

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("Start", disabled=st.session_state.running)
stop_btn = col2.button("Stop", disabled=not st.session_state.running)

if start_btn and not st.session_state.running:
    if not selected_model:
        st.sidebar.error("Please add a YOLO .pt model to the models/ folder first.")
    else:
        st.session_state.running = True
        st.session_state.stop_event = threading.Event()
        st.session_state.out_queue = queue.Queue(maxsize=2)
        st.session_state.pipeline_thread = Pipeline(config, st.session_state.out_queue, st.session_state.stop_event)
        st.session_state.pipeline_thread.start()
        st.rerun()

if stop_btn and st.session_state.running:
    st.session_state.running = False
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    # Let it spin down naturally
    st.rerun()

# ---- MAIN LAYOUT ----
st.title(f"Traffic Density Estimator {'🟢 Running' if st.session_state.running else '🔴 Stopped'}")

tab_dashboard, tab_analytics = st.tabs(["Dashboard", "Analytics"])

with tab_dashboard:
    col_video, col_gauges = st.columns([2, 1])
    
    with col_video:
        video_placeholder = st.empty()
        alert_placeholder = st.empty()
        
    with col_gauges:
        gauges_placeholder = st.empty()

    if st.session_state.running and st.session_state.out_queue is not None:
        while st.session_state.running:
            try:
                data = st.session_state.out_queue.get(timeout=0.1)
                
                if "error" in data:
                    st.error(data["error"])
                    st.session_state.running = False
                    break
                if "status" in data and data["status"] == "stopped":
                    st.session_state.running = False
                    break
                    
                if "frame_jpg" in data:
                    video_placeholder.image(data["frame_jpg"], use_container_width=True)
                    
                if "zone_states" in data:
                    with gauges_placeholder.container():
                        for zid, zstate in data["zone_states"].items():
                            # Map levels to emoji/colors for visual pop
                            level = zstate["level"]
                            color_indicator = "🟢" if level == "LOW" else "🟡" if level == "MEDIUM" else "🔴"
                            
                            st.metric(
                                label=f"{color_indicator} {zstate['name']}",
                                value=f"{zstate['count']} vehicles",
                                delta=level,
                                delta_color="inverse" if level == "HIGH" else "off"
                            )
                            
                if "active_alerts" in data and len(data["active_alerts"]) > 0:
                    alert_placeholder.error(f"🚨 ACTIVE ALERTS IN: {', '.join(data['active_alerts'])}")
                    # Play sound (Streamlit audio hack)
                    if len(data.get("triggered_alerts", [])) > 0:
                         if os.path.exists(config.get("alert_sound", "")):
                             pass # Ideally inject a hidden audio tag or use st.audio(..., autoplay=True)
                             # st.audio(config["alert_sound"], autoplay=True) # Streamlit autoplay can be finicky
                else:
                    alert_placeholder.empty()

            except queue.Empty:
                time.sleep(0.05)
                
            # Allow Streamlit to respond to Stop button
            if not st.session_state.running:
                break

with tab_analytics:
    st.header("Historical Analysis")
    if st.button("Refresh Analytics"):
        if os.path.exists(config["log_path"]):
            col_a, col_b = st.columns(2)
            with col_a:
                fig_peak = generate_peak_hours_chart(config["log_path"])
                if fig_peak:
                    st.plotly_chart(fig_peak, use_container_width=True)
            with col_b:
                fig_heat = generate_session_heatmap_chart(config["log_path"])
                if fig_heat:
                    st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No log data available yet.")
