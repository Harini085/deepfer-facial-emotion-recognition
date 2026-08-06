"""
DeepFER — Shared Configuration
===============================
Central place for paths, class names, and inference settings used by
predict.py, model_loader.py, utils.py, streamlit_app.py, and realtime_video.py.
"""

import os
import json

# --- Paths ---
# Adjust these if your folder layout differs. By default this expects:
#   models/best_model.keras
#   models/class_names.pkl
#   models/model_config.json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
MODEL_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

# --- Fallback class names (used only if class_names.pkl isn't found) ---
DEFAULT_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Emoji per class, purely cosmetic for the Streamlit UI
CLASS_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

# --- Inference settings ---
DEFAULT_INPUT_SIZE = 48  # overridden automatically from model_config.json if present
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"  # bundled with opencv-python's cv2.data

# Real-time video smoothing
MAJORITY_VOTE_WINDOW = 8  # number of recent frames to smooth predictions over

def load_model_config():
    """Load input size / class names saved during training, with sane fallbacks."""
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg.get("input_size", DEFAULT_INPUT_SIZE), cfg.get("class_names", DEFAULT_CLASS_NAMES)
    return DEFAULT_INPUT_SIZE, DEFAULT_CLASS_NAMES
