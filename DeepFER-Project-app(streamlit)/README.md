# 😊 DeepFER — Facial Expression Recognition

An end-to-end deep learning project for recognizing facial expressions from FER2013:
data pipeline, 8-architecture model comparison, explainability-ready evaluation,
real-time video inference, and a deployable Streamlit app.

---

## 📋 Project Overview

DeepFER classifies faces into 7 emotions — **angry, disgust, fear, happy, neutral, sad, surprise**
— from 48×48 grayscale images. The project covers the full pipeline: EDA, preprocessing/augmentation
(including CutMix and MixUp), training and comparing 8 architectures (6 ImageNet-pretrained CNNs +
2 from-scratch transformers), a full evaluation suite, and deployment via a Streamlit web app with
image, video, and live-webcam support.

## 📂 Dataset

**FER2013**: 35,887 images (28,709 train / 7,178 test), 48×48 grayscale, 7 classes.
Notably imbalanced — `happy` accounts for ~25% of the data, `disgust` only ~1.5%.
Get it from Kaggle (e.g. the `msambare/fer2013` dataset) and add it as a Kaggle notebook input,
or place it locally under `dataset/train/<class>/` and `dataset/test/<class>/`.

## 🔍 EDA & 🏋️ Training

`notebooks/DeepFER_Combined.ipynb` — a single notebook covering the full pipeline in one run:

**Part 1 — EDA:** data loading + 18 visualizations covering class distribution, sample grids,
image geometry, pixel/brightness/contrast/entropy statistics, edge density, average class images,
PCA/t-SNE projections, a feature correlation heatmap, and data-quality checks (corrupted, duplicate,
blurry, and outlier images). Every chart includes an interpretation of what it shows.

**Key findings:** no corrupted images; ~1,850 duplicate files; significant class imbalance;
`fear`, `sad`, and `neutral` are the most visually confusable classes.

**Part 2 — Training:** preprocessing, augmentation, and training all 8 architectures:

| Type | Architectures |
|---|---|
| Pretrained CNN (transfer learning) | EfficientNetB0, EfficientNetB3, ResNet50, DenseNet121, MobileNetV3Small, ConvNeXtTiny |
| From-scratch Transformer | Vision Transformer (ViT), Swin Transformer (simplified) |

Augmentation includes flip, rotation, brightness/contrast, Gaussian noise, random crop, CutMix,
and MixUp. Training uses EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and class weighting to
counter FER2013's imbalance.

Evaluation reports accuracy, precision, recall, F1, top-5 accuracy, ROC/PR curves, confusion
matrices, classification reports, inference latency, and model size — collected into a ranked
comparison table, with the best model (by macro-F1) saved automatically.

> ⚠️ **Run this on Kaggle with GPU + internet enabled.** Training all 8 architectures to full
> convergence is a multi-hour job. Set `FAST_DEV_RUN = True` first to sanity-check the whole
> notebook in minutes, or trim `MODELS_TO_RUN` to fit your session's GPU quota.

## 📊 Results

After training, `model_comparison.csv` in your output directory holds the full leaderboard, and
`best_model.keras` is the top performer by macro-F1. (Populate this section with your own numbers
once training completes — results depend on how long each architecture is trained.)

## 🚀 Deployment

### App structure
```
app/
├── config.py          # paths, class names, settings
├── custom_layers.py   # ViT/Swin layer definitions (needed to reload those models)
├── model_loader.py     # cached model loading
├── utils.py           # preprocessing, face detection, majority-vote smoothing
├── predict.py         # single image / folder / uploaded image / video frame inference
├── realtime_video.py   # webcam / video file real-time detection (OpenCV)
└── streamlit_app.py    # the web app
```

### Running inference from the command line
```bash
python app/predict.py path/to/image.jpg
python app/predict.py path/to/folder/
```

### Real-time video
```bash
python app/realtime_video.py                          # webcam
python app/realtime_video.py --source video.mp4 --output annotated.mp4
```

### Streamlit app
```bash
streamlit run app/streamlit_app.py
```
Features: image upload, video upload with per-frame analysis, live webcam (via `streamlit-webrtc`),
prediction confidence and probability charts, prediction history, and a downloadable CSV report.
Uses the saved model only — no retraining happens in the app.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ☁️ Deploy to Streamlit Community Cloud

1. Push this whole folder to a GitHub repo (public, or private on a paid Streamlit plan).
   `models/` (11 MB) and `app/haarcascade_frontalface_default.xml` are small enough to commit
   directly — no Git LFS needed. `dataset/` is git-ignored; it isn't needed for the deployed app.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo/branch, and set **Main file path** to `app/streamlit_app.py`.
4. Deploy. `requirements.txt` and `runtime.txt` at the repo root are picked up automatically.
5. First load will be slower (\~1–2 min) while TensorFlow installs and the model loads; after
   that `@st.cache_resource` keeps the model warm.

**Notes specific to Cloud hosting:**
- `requirements.txt` uses `opencv-python-headless` and `tensorflow-cpu` — the regular
  `opencv-python` package fails on Streamlit Cloud (`ImportError: libGL.so.1`) because the
  container has no display libraries.
- The **Live Webcam** tab needs a STUN server to punch through NAT on Cloud (`localhost` doesn't
  need this, which is why it can look fine locally and fail once deployed) — already configured
  in `streamlit_app.py` via `rtc_configuration`.
- Streamlit Cloud's free tier has ~1 GB RAM. This app's placeholder model is small (~10 MB), but
  if you swap in one of the bigger EfficientNet/ConvNeXt backbones after full training, watch
  memory during the video/webcam modes, which hold frames in memory per request.

> ⚠️ The bundled `models/best_model.keras` is the **2-epoch placeholder** described in
> `PLACEHOLDER_NOTICE.md` — the app deploys and runs fine, but predictions are near-random until
> you replace it with a fully-trained model from `notebooks/DeepFER_Combined.ipynb`.

## 🗂️ Project Structure

```
DeepFER-Project/
├── notebooks/
│   └── DeepFER_Combined.ipynb
├── models/
│   ├── best_model.keras
│   ├── class_names.pkl
│   ├── label_encoder.pkl
│   ├── model_config.json
│   └── training_history.pkl
├── app/
│   ├── config.py
│   ├── custom_layers.py
│   ├── model_loader.py
│   ├── utils.py
│   ├── predict.py
│   ├── realtime_video.py
│   └── streamlit_app.py
├── dataset/
│   ├── train/
│   └── test/
├── train.py
├── requirements.txt
└── README.md
```

## 🔮 Future Work

- Grad-CAM / SHAP explainability overlays for individual predictions
- Model quantization (TFLite/ONNX) for faster mobile/edge inference
- Fine-tuning the pretrained CNN backbones (currently frozen) for a possible accuracy gain
- Expanding the Swin implementation toward the full shifted-window architecture with more stages
- Deploying the Streamlit app to Streamlit Cloud or Hugging Face Spaces

## 📚 References

- Goodfellow, I. et al. "Challenges in Representation Learning: A report on three machine learning
  contests." (FER2013 origin)
- Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
- Liu, Z. et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
