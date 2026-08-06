"""
DeepFER — Shared Utilities
============================
Preprocessing, face detection, and plotting helpers shared by predict.py,
realtime_video.py, and streamlit_app.py.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

from config import FACE_CASCADE_PATH


def preprocess_face(face_img, input_size):
    """Take a BGR or grayscale face crop (numpy array) and prepare it for the model:
    resize to input_size, convert to 3-channel RGB float32 in [0, 255] (matching training)."""
    if face_img.ndim == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[-1] == 1:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[-1] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    face_img = cv2.resize(face_img, (input_size, input_size))
    face_img = face_img.astype("float32")
    return face_img


def get_face_detector():
    """Return an OpenCV Haar cascade face detector.
    Prefers the copy bundled directly in this app/ folder (haarcascade_frontalface_default.xml) —
    on some Windows OpenCV installs, cv2.data.haarcascades points to a folder that's missing its
    XML files, so we don't rely on that alone."""
    bundled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FACE_CASCADE_PATH)
    fallback_path = cv2.data.haarcascades + FACE_CASCADE_PATH

    cascade_path = bundled_path if os.path.exists(bundled_path) else fallback_path
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(
            f"Could not load face cascade. Tried:\n  {bundled_path}\n  {fallback_path}\n"
            "Make sure haarcascade_frontalface_default.xml is present in the app/ folder."
        )
    return detector


def detect_faces(frame_bgr, detector, scale_factor=1.1, min_neighbors=5, min_size=(48, 48)):
    """Detect faces in a BGR frame. Returns a list of (x, y, w, h) bounding boxes."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    return faces


def predict_array(model, class_names, input_size, face_img):
    """Run inference on a single preprocessed-or-raw face crop.
    Returns (predicted_class, confidence, all_probabilities)."""
    processed = preprocess_face(face_img, input_size)
    batch = np.expand_dims(processed, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), probs


def top_k_predictions(probs, class_names, k=5):
    """Return the top-k (class_name, probability) pairs, sorted descending."""
    k = min(k, len(class_names))
    idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idx]


class MajorityVoteSmoother:
    """Smooths per-frame predictions over a sliding window using majority vote —
    reduces flicker in real-time video classification."""

    def __init__(self, window_size=8):
        self.window_size = window_size
        self.history = []

    def update(self, predicted_class):
        self.history.append(predicted_class)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        # Majority vote over the current window
        values, counts = np.unique(self.history, return_counts=True)
        return values[np.argmax(counts)]

    def reset(self):
        self.history = []
