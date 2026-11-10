"""Video-level deepfake detection by scoring sampled frames.

Aggregation: every sampled frame gets a P(fake) score from the image model and
the video score is the mean of those scores; the verdict is FAKE when that mean
reaches the threshold. This is frame averaging, not a temporal model.
"""

import cv2
import numpy as np

from predict import DEFAULT_MODEL, load_detector


def _prepare(frame, size):
    """Convert a BGR frame to RGB and resize it for the model."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def predict_video(path, model, stride=15, max_frames=60, threshold=0.5):
    """Score up to `max_frames` frames, sampling one out of every `stride` frames."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    size = model.input_shape[1]
    frames, position = [], 0
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if position % stride == 0:
                frames.append(_prepare(frame, size))
            position += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No readable frames in {path}")

    scores = model.predict(np.stack(frames).astype("float32") / 255.0, verbose=0).ravel()
    mean_score = float(scores.mean())
    return {
        "label": "FAKE" if mean_score >= threshold else "REAL",
        "fake_probability": mean_score,
        "frames_analysed": len(scores),
        "frame_scores": scores,
    }


def add_arguments(parser):
    parser.add_argument("video", help="video file to analyse")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--stride", type=int, default=15, help="analyse 1 of N frames")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)


def run(args):
    result = predict_video(
        args.video,
        load_detector(args.model),
        stride=max(1, args.stride),
        max_frames=max(1, args.max_frames),
        threshold=args.threshold,
    )
    scores = result["frame_scores"]
    print(f"Video: {args.video}")
    print(f"Frames analysed: {result['frames_analysed']} (stride {args.stride})")
    print(f"Mean fake probability: {result['fake_probability']:.3f}")
    print(f"Per-frame range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"Verdict: {result['label']}")
