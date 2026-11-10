"""Train the DeepFakeShield CNN on dataset/real and dataset/fake."""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks

from data import load_dataset, split_by_group
from model import build_model

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "deepfake_cnn.keras"
DEFAULT_PLOT = Path(__file__).resolve().parents[1] / "screenshots" / "training_curves.png"


def add_arguments(parser):
    parser.add_argument("--dataset", default="dataset", help="folder with real/ and fake/")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="output model path")
    parser.add_argument("--plot", default=str(DEFAULT_PLOT), help="output training curve path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)


def _plot_history(history, output_path):
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(history.history[metric], label=f"train {metric}")
        axis.plot(history.history[f"val_{metric}"], label=f"validation {metric}")
        axis.set_title(metric.capitalize())
        axis.set_xlabel("epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Training curves saved to {output_path}")


def run(args):
    tf.keras.utils.set_random_seed(args.seed)

    images, labels, groups = load_dataset(args.dataset, args.img_size)
    images = images.astype("float32") / 255.0
    train, val, test = split_by_group(
        images, labels, groups, args.val_size, args.test_size, args.seed
    )

    for name, (X, y) in zip(("train", "validation", "test"), (train, val, test)):
        print(f"{name:10s} images={len(X):5d}  real={int((y == 0).sum()):5d}  fake={int((y == 1).sum()):5d}")

    model = build_model(args.img_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train[1])

    history = model.fit(
        *train,
        validation_data=val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=dict(enumerate(weights)),
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                model_path, monitor="val_loss", save_best_only=True, verbose=1
            ),
        ],
        verbose=2,
    )

    X_test, y_test = test
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
    print(f"\nTest loss {loss:.4f} | accuracy {accuracy:.4f}")
    if len(np.unique(y_test)) < 2:
        print("Warning: the test split contains a single class; some metrics are undefined.")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["real", "fake"],
            digits=3,
            zero_division=0,
        )
    )
    print(f"Model saved to {model_path}")
    _plot_history(history, args.plot)
