"""
DeepFER — Streamlit Application
==================================
Features: upload image, upload video, live webcam (via streamlit-webrtc), prediction
confidence, prediction history, charts, dark theme, downloadable prediction report.

Uses the saved best_model.keras only — no retraining happens here.

Run with:  streamlit run streamlit_app.py
"""

import os
import io
import time
import json
import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model_loader import get_everything
from predict import predict_uploaded_image, predict_frame
from utils import get_face_detector
from config import CLASS_EMOJIS


# -------------------------------------------------------------------------
# Page config + dark theme styling
# -------------------------------------------------------------------------
st.set_page_config(page_title="DeepFER — Facial Expression Recognition", page_icon="😊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px; padding: 1.2rem; border: 1px solid #2d3748;
    }
    .big-emoji { font-size: 3.5rem; text-align: center; }
    h1, h2, h3 { color: #f3f4f6 !important; }
    .stButton > button {
        background-color: #6366f1; color: white; border-radius: 8px; border: none;
    }
    .stButton > button:hover { background-color: #4f46e5; }
</style>
""", unsafe_allow_html=True)

st.title("😊 DeepFER — Facial Expression Recognition")
st.caption("Upload an image or video, or use your webcam, to detect facial expressions in real time.")

# -------------------------------------------------------------------------
# Session state for prediction history
# -------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, source, predicted_class, confidence


def log_prediction(source, predicted_class, confidence):
    st.session_state.history.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "predicted_class": predicted_class,
        "confidence": confidence,
    })


# -------------------------------------------------------------------------
# Load model once (cached)
# -------------------------------------------------------------------------
@st.cache_resource
def _load():
    return get_everything()

try:
    model, class_names, input_size = _load()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(str(e))

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Input source", ["📷 Upload Image", "🎞️ Upload Video", "🎥 Live Webcam"])
    st.divider()
    st.header("📜 Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, height=200, use_container_width=True)
        csv_buffer = io.StringIO()
        hist_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"deepfer_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet.")


def render_probability_chart(probabilities: dict):
    df = pd.DataFrame({"Emotion": list(probabilities.keys()), "Probability": list(probabilities.values())})
    df = df.sort_values("Probability", ascending=True)
    fig = px.bar(
        df, x="Probability", y="Emotion", orientation="h",
        color="Probability", color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_result(result, source_label):
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    emoji = CLASS_EMOJIS.get(pred_class, "🙂")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'<div class="big-emoji">{emoji}</div>', unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center'>{pred_class.title()}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:1.2rem;'>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
    with col2:
        st.subheader("Class Probabilities")
        render_probability_chart(result["probabilities"])

    log_prediction(source_label, pred_class, confidence)


# -------------------------------------------------------------------------
# Mode: Upload Image
# -------------------------------------------------------------------------
if mode == "📷 Upload Image" and model_loaded:
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        col_img, col_result = st.columns([1, 1.4])
        with col_img:
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Analyzing expression..."):
            result = predict_uploaded_image(pil_img)
        with col_result:
            render_result(result, source_label=uploaded_file.name)

# -------------------------------------------------------------------------
# Mode: Upload Video
# -------------------------------------------------------------------------
elif mode == "🎞️ Upload Video" and model_loaded:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tmp_path = f"/tmp/{uploaded_video.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(tmp_path)
        sample_every_n = st.slider("Sample every Nth frame (higher = faster)", 5, 60, 15)

        if st.button("▶️ Run Analysis"):
            cap = cv2.VideoCapture(tmp_path)
            detector = get_face_detector()
            frame_results, frame_idx = [], 0
            progress = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every_n == 0:
                    faces = predict_frame(frame, detector)
                    for f in faces:
                        frame_results.append({"frame": frame_idx, **f})
                frame_idx += 1
                progress.progress(min(frame_idx / total_frames, 1.0))
            cap.release()

            if frame_results:
                df = pd.DataFrame(frame_results)
                st.success(f"Analyzed {len(df)} face detections across {frame_idx} frames.")
                counts = df["predicted_class"].value_counts().reset_index()
                counts.columns = ["Emotion", "Count"]
                fig = px.pie(counts, names="Emotion", values="Count", template="plotly_dark",
                             title="Emotion Distribution Across Video")
                st.plotly_chart(fig, use_container_width=True)

                dominant = counts.iloc[0]["Emotion"]
                log_prediction(uploaded_video.name, dominant, df["confidence"].mean())
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No faces detected in the sampled frames.")

# -------------------------------------------------------------------------
# Mode: Live Webcam (streamlit-webrtc)
# -------------------------------------------------------------------------
elif mode == "🎥 Live Webcam" and model_loaded:
    st.info("Live webcam requires the `streamlit-webrtc` package (see requirements.txt) and browser camera permission.")
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        detector = get_face_detector()

        class EmotionProcessor(VideoProcessorBase):
            def __init__(self):
                self.last_result = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                faces = predict_frame(img, detector)
                for f in faces:
                    x, y, w, h = f["bbox"]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (46, 204, 113), 2)
                    label = f"{f['predicted_class']} ({f['confidence']:.0%})"
                    cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    self.last_result = f
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="deepfer-webcam",
            video_processor_factory=EmotionProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        if ctx.video_processor and ctx.video_processor.last_result:
            r = ctx.video_processor.last_result
            log_prediction("webcam", r["predicted_class"], r["confidence"])
    except ImportError:
        st.error("`streamlit-webrtc` is not installed. Run: pip install streamlit-webrtc av")

# -------------------------------------------------------------------------
if not model_loaded:
    st.warning("⚠️ No trained model found. Run the Stage 2 training notebook first, "
               "then place `best_model.keras`, `class_names.pkl`, and `model_config.json` in the `models/` folder.")
