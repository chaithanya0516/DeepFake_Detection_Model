"""CNN architecture for real/fake image classification."""

from tensorflow.keras import layers, models


def build_model(img_size=128):
    """A small CNN with a single sigmoid output holding P(fake)."""
    return models.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="deepfake_cnn",
    )
