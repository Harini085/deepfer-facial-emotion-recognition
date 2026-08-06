"""
DeepFER — Inference Functions
==============================
Loads the saved best_model.keras (no retraining) and exposes prediction functions for:
  - a single image file
  - a folder of images
  - an already-loaded uploaded image (PIL/numpy)
  - a single video/webcam frame (numpy array, BGR)

Each function returns predicted class, confidence, and top-5 probabilities.
Run as a script for a quick CLI demo:  python predict.py path/to/image.jpg
"""

import os
import sys
import glob
import time

import cv2
import numpy as np
from PIL import Image

from model_loader import get_everything
from utils import get_face_detector, detect_faces, predict_array, top_k_predictions


def predict_image(image_path, use_face_detection=True):
    """Predict emotion for a single image file on disk.
    If use_face_detection is True, detects the largest face first; otherwise
    classifies the whole image (useful for pre-cropped FER2013-style inputs)."""
    model, class_names, input_size = get_everything()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    face_crop = img_bgr
    bbox = None
    if use_face_detection:
        detector = get_face_detector()
        faces = detect_faces(img_bgr, detector)
        if len(faces) > 0:
            # Use the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = img_bgr[y:y + h, x:x + w]
            bbox = (int(x), int(y), int(w), int(h))

    pred_class, confidence, probs = predict_array(model, class_names, input_size, face_crop)
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top5": top_k_predictions(probs, class_names, k=5),
        "bbox": bbox,
        "probabilities": dict(zip(class_names, probs.tolist())),
    }


def predict_folder(folder_path, extensions=(".jpg", ".jpeg", ".png")):
    """Predict emotions for every image in a folder. Returns a list of per-file results."""
    results = []
    files = sorted([
        f for f in glob.glob(os.path.join(folder_path, "**", "*"), recursive=True)
        if f.lower().endswith(extensions)
    ])
    for f in files:
        try:
            result = predict_image(f, use_face_detection=True)
            result["filepath"] = f
            results.append(result)
        except Exception as e:
            results.append({"filepath": f, "error": str(e)})
    return results


def predict_uploaded_image(pil_image, use_face_detection=True):
    """Predict emotion for an in-memory PIL image (e.g. from a Streamlit file_uploader)."""
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    model, class_names, input_size = get_everything()
    face_crop = img_bgr
    bbox = None
    if use_face_detection:
        detector = get_face_detector()
        faces = detect_faces(img_bgr, detector)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = img_bgr[y:y + h, x:x + w]
            bbox = (int(x), int(y), int(w), int(h))

    pred_class, confidence, probs = predict_array(model, class_names, input_size, face_crop)
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top5": top_k_predictions(probs, class_names, k=5),
        "bbox": bbox,
        "probabilities": dict(zip(class_names, probs.tolist())),
    }


def predict_frame(frame_bgr, detector=None):
    """Predict emotion(s) for a single video/webcam frame (numpy array, BGR).
    Returns a list of per-face results, each with a bounding box — supports multiple
    faces in one frame. Pass a pre-built detector to avoid reloading the cascade every call."""
    model, class_names, input_size = get_everything()
    if detector is None:
        detector = get_face_detector()

    faces = detect_faces(frame_bgr, detector)
    results = []
    for (x, y, w, h) in faces:
        face_crop = frame_bgr[y:y + h, x:x + w]
        pred_class, confidence, probs = predict_array(model, class_names, input_size, face_crop)
        results.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "predicted_class": pred_class,
            "confidence": confidence,
            "top5": top_k_predictions(probs, class_names, k=5),
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    start = time.time()
    if os.path.isdir(target):
        results = predict_folder(target)
        for r in results:
            if "error" in r:
                print(f"❌ {r['filepath']}: {r['error']}")
            else:
                print(f"✅ {r['filepath']}: {r['predicted_class']} ({r['confidence']:.1%})")
    else:
        result = predict_image(target)
        print(f"Predicted: {result['predicted_class']} ({result['confidence']:.1%})")
        print("Top-5:", result["top5"])
    print(f"\nDone in {time.time() - start:.2f}s")
