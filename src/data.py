"""Dataset loading and leakage-safe splitting.

Frames extracted from the same source video belong to the same *group*, so the
train/validation/test split is done per group instead of per image.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

IMG_SIZE = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = ("real", "fake")  # label 0 = real, label 1 = fake


def load_image(path, size=IMG_SIZE):
    """Read an image as an RGB uint8 array resized to (size, size)."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable or unsupported image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def source_id(path):
    """Group id for a frame: 'video01_frame007.jpg' -> 'video01', 'face.png' -> 'face'."""
    stem = Path(path).stem
    return stem.rsplit("_frame", 1)[0]


def load_dataset(dataset_dir, size=IMG_SIZE):
    """Load dataset/real and dataset/fake.

    Returns (images, labels, groups) as RGB uint8 arrays / ints / strings.
    """
    dataset_dir = Path(dataset_dir)
    images, labels, groups, unreadable = [], [], [], 0

    for label, category in enumerate(CATEGORIES):
        folder = dataset_dir / category
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder.resolve()}")
        for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            try:
                images.append(load_image(path, size))
            except ValueError:
                unreadable += 1
                continue
            labels.append(label)
            groups.append(f"{category}/{source_id(path)}")

    if not images:
        raise ValueError(
            f"No images found in {dataset_dir.resolve()}/real or /fake. "
            "See README.md for the dataset layout."
        )
    if unreadable:
        print(f"Warning: skipped {unreadable} unreadable file(s).")

    return np.array(images), np.array(labels), np.array(groups)


def split_by_group(
    images, labels, groups, val_size=0.15, test_size=0.15, seed=42, attempts=20
):
    """Split per source video so frames of one video never span two splits.

    Seeds are retried until every split holds both classes, so the reported
    metrics are meaningful; the last attempt is used if that is not possible.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(
            f"Only {len(unique_groups)} distinct source group(s) found; at least 3 are "
            "needed for a train/validation/test split. Name frames after their source "
            "video, e.g. video01_frame001.jpg."
        )

    for offset in range(attempts):
        train_idx, val_idx, test_idx = _split_once(
            images, labels, groups, val_size, test_size, seed + offset
        )
        if all(len(np.unique(labels[idx])) == 2 for idx in (train_idx, val_idx, test_idx)):
            break
    else:
        print("Warning: a split contains a single class; metrics will be incomplete.")

    split = lambda idx: (images[idx], labels[idx])
    return split(train_idx), split(val_idx), split(test_idx)


def _split_once(images, labels, groups, val_size, test_size, seed):
    """One group-shuffled train/validation/test split by index."""
    index = np.arange(len(images))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(index, labels, groups))

    val_ratio = val_size / (1.0 - test_size)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_rel, val_rel = next(
        splitter.split(train_val_idx, labels[train_val_idx], groups[train_val_idx])
    )
    return train_val_idx[train_rel], train_val_idx[val_rel], test_idx
