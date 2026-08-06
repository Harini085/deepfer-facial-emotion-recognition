"""
DeepFER — Model Loader
=======================
Loads the saved best_model.keras (and its class names / input size) exactly once,
so predict.py, streamlit_app.py, and realtime_video.py can all share a single
cached model instance instead of re-loading it on every call.
"""

import os
import pickle
import functools

import tensorflow as tf

import custom_layers  # noqa: F401 — import registers AddClsToken/SwinBlock/PatchMerging for deserialization
from config import MODEL_PATH, CLASS_NAMES_PATH, load_model_config


@functools.lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained Keras model. Raises a clear error if the file is missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find a trained model at '{MODEL_PATH}'. "
            "Run the Stage 2 training notebook first, or copy best_model.keras into the models/ folder."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@functools.lru_cache(maxsize=1)
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "rb") as f:
            return pickle.load(f)
    _, class_names = load_model_config()
    return class_names


@functools.lru_cache(maxsize=1)
def get_input_size():
    size, _ = load_model_config()
    return size


def get_everything():
    """Convenience accessor: (model, class_names, input_size)."""
    return load_model(), load_class_names(), get_input_size()
