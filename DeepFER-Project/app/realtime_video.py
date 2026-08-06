"""
DeepFER — Real-Time Video Detection
=====================================
Runs live facial expression detection on a webcam feed or an uploaded video file,
using OpenCV for capture/display and the saved best_model.keras for inference (no retraining).

For each detected face, draws:
  - bounding box
  - predicted emotion (majority-vote smoothed over recent frames, to reduce flicker)
  - confidence
  - current FPS
  - frame number

Usage:
    python realtime_video.py                  # webcam (device 0)
    python realtime_video.py --source 1        # a different webcam index
    python realtime_video.py --source video.mp4  # a video file
    python realtime_video.py --source video.mp4 --output annotated.mp4  # save the result

Press 'q' to quit an interactive (webcam) session.
"""

import argparse
import time

import cv2
import numpy as np

from model_loader import get_everything
from utils import get_face_detector, detect_faces, predict_array, MajorityVoteSmoother
from config import MAJORITY_VOTE_WINDOW, CLASS_EMOJIS


BOX_COLOR = (46, 204, 113)     # green (BGR)
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (46, 125, 50)


def draw_annotation(frame, bbox, label, confidence, fps, frame_num):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

    caption = f"{label} ({confidence:.0%})"
    (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - text_h - 12), (x + text_w + 8, y), BG_COLOR, -1)
    cv2.putText(frame, caption, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def run(source=0, output_path=None, display=True, max_frames=None):
    model, class_names, input_size = get_everything()
    detector = get_face_detector()

    # Per-face-slot smoothers, keyed by an approximate face position bucket so multiple
    # faces in frame don't smear votes together. Simple and effective for typical use.
    smoothers = {}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    frame_num = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if max_frames and frame_num > max_frames:
            break

        faces = detect_faces(frame, detector)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            pred_class, confidence, probs = predict_array(model, class_names, input_size, face_crop)

            # Bucket faces by rough grid position so smoothing tracks "the same" face across frames
            bucket = (x // 50, y // 50)
            if bucket not in smoothers:
                smoothers[bucket] = MajorityVoteSmoother(window_size=MAJORITY_VOTE_WINDOW)
            smoothed_label = smoothers[bucket].update(pred_class)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_annotation(frame, (x, y, w, h), smoothed_label, confidence, fps, frame_num)

        if writer:
            writer.write(frame)
        if display:
            cv2.imshow("DeepFER — Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFER real-time video/webcam facial expression detection")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--output", default=None, help="Optional path to save the annotated video")
    parser.add_argument("--no-display", action="store_true", help="Disable the live preview window (headless)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source=source, output_path=args.output, display=not args.no_display, max_frames=args.max_frames)
