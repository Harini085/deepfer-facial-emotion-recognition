"""
DeepFER — CLI Training Script
================================
A scriptable wrapper around the same training pipeline used in the Stage 2 notebook,
for automated or headless runs (e.g. `python train.py --models EfficientNetB0 ViT_Scratch --epochs 30`).

For interactive exploration, EDA, and rich inline visualizations, use the notebooks instead —
this script is meant for reproducible, unattended training runs.
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 7
IMG_SIZE_NATIVE = 48
RANDOM_STATE = 42

CNN_BACKBONES = {
    "EfficientNetB0": (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input, 224),
    "EfficientNetB3": (keras.applications.EfficientNetB3, keras.applications.efficientnet.preprocess_input, 300),
    "ResNet50": (keras.applications.ResNet50, keras.applications.resnet.preprocess_input, 224),
    "DenseNet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input, 224),
    "MobileNetV3Small": (keras.applications.MobileNetV3Small, keras.applications.mobilenet_v3.preprocess_input, 224),
    "ConvNeXtTiny": (keras.applications.ConvNeXtTiny, keras.applications.convnext.preprocess_input, 224),
}


def build_cnn_backbone(name, freeze_base=True):
    ctor, preprocess_fn, img_size = CNN_BACKBONES[name]
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = preprocess_fn(inputs)
    base = ctor(include_top=False, weights="imagenet", input_tensor=x, pooling="avg")
    base.trainable = not freeze_base
    x = layers.Dropout(0.3)(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name), img_size


def get_model_and_size(name):
    if name in ("ViT_Scratch", "Swin_Scratch"):
        # Import the shared architecture definitions used by the Stage 2 notebook / app
        # to keep a single source of truth for these custom layers.
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
        import custom_layers  # noqa: F401 registers the layers
        raise NotImplementedError(
            "ViT_Scratch / Swin_Scratch builders live in the Stage 2 notebook for now. "
            "Run the notebook for these two architectures, or copy the build_vit()/build_swin() "
            "functions from notebooks/DeepFER_Stage2_Training.ipynb into this file if you want "
            "them available from the CLI too."
        )
    return build_cnn_backbone(name)


def find_data_dir():
    candidates = ["/kaggle/input/fer2013", "dataset", "../dataset"]
    for d in candidates:
        if os.path.isdir(os.path.join(d, "train")) and os.path.isdir(os.path.join(d, "test")):
            return d
    for cand in glob.glob("/kaggle/input/*"):
        if os.path.isdir(os.path.join(cand, "train")) and os.path.isdir(os.path.join(cand, "test")):
            return cand
    raise FileNotFoundError("Could not auto-detect dataset directory. Pass --data-dir explicitly.")


def main():
    parser = argparse.ArgumentParser(description="DeepFER CLI training script")
    parser.add_argument("--models", nargs="+", default=["MobileNetV3Small"],
                         help="Which architectures to train, e.g. --models EfficientNetB0 ResNet50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    data_dir = args.data_dir or find_data_dir()
    train_dir, test_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "test")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📂 Data dir: {data_dir}")

    for model_name in args.models:
        print(f"\n{'='*60}\n🏋️ Training {model_name}\n{'='*60}")
        model, img_size = get_model_and_size(model_name)

        train_ds = keras.utils.image_dataset_from_directory(
            train_dir, image_size=(img_size, img_size), color_mode="grayscale",
            batch_size=args.batch_size, validation_split=0.15, subset="training", seed=RANDOM_STATE,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            train_dir, image_size=(img_size, img_size), color_mode="grayscale",
            batch_size=args.batch_size, validation_split=0.15, subset="validation", seed=RANDOM_STATE,
        )
        class_names = train_ds.class_names

        def to_rgb(img, label):
            return tf.image.grayscale_to_rgb(img), tf.one_hot(label, NUM_CLASSES)

        train_ds = train_ds.map(to_rgb).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(to_rgb).prefetch(tf.data.AUTOTUNE)

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        ckpt_path = os.path.join(args.output_dir, f"{model_name}_best.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]

        start = time.time()
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
        print(f"✅ {model_name} trained in {time.time() - start:.1f}s -> {ckpt_path}")

        with open(os.path.join(args.output_dir, "class_names.pkl"), "wb") as f:
            pickle.dump(class_names, f)
        with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
            json.dump({"best_model_name": model_name, "input_size": img_size, "class_names": class_names}, f, indent=2)


if __name__ == "__main__":
    main()
