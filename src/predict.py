"""Classify an image as REAL or FAKE with a trained model."""

import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

from data import load_image

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "deepfake_cnn.keras"


def load_detector(model_path=DEFAULT_MODEL):
    """Load a saved model, with a clear error when training has not run yet."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python src/main.py train"
        )
    return load_model(path)


def predict_image(image, model):
    """Return (label, P(fake)) for an image path or an RGB uint8 array."""
    size = model.input_shape[1]
    if not isinstance(image, np.ndarray):
        image = load_image(image, size)
    batch = image.astype("float32")[None] / 255.0
    fake_probability = float(model.predict(batch, verbose=0).ravel()[0])
    return ("FAKE" if fake_probability >= 0.5 else "REAL"), fake_probability


def add_arguments(parser):
    parser.add_argument("images", nargs="+", help="image files to classify")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))


def run(args):
    model = load_detector(args.model)
    for image_path in args.images:
        try:
            label, score = predict_image(image_path, model)
        except ValueError as error:
            print(f"{image_path}: skipped ({error})")
            continue
        print(f"{image_path}: {label} (fake probability {score:.3f})")
