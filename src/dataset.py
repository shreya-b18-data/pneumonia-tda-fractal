"""Index the Kaggle chest X-ray folders and load preprocessed grayscale images."""
import re

import cv2
import numpy as np

from config import CLASSES, DATA_DIR, IMG_SIZE


def index_split(split, root=DATA_DIR):
    """Return (paths, labels) for split in {'train','val','test'}."""
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        for f in sorted((root / split / cls).glob("*")):
            if f.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                paths.append(f)
                labels.append(label)
    return paths, np.array(labels)


def patient_id(path):
    """Best-effort patient id from the Kaggle file names, used to avoid leakage in cross-validation.
    PNEUMONIA: person123_bacteria_456.jpeg -> 'p123'.  NORMAL: IM-0115-0001.jpeg -> 'n0115'."""
    name = path.name
    m = re.match(r"person(\d+)_", name)
    if m:
        return f"p{m.group(1)}"
    m = re.search(r"IM-(\d+)", name)
    if m:
        return f"n{m.group(1)}"
    return name  # unknown pattern: treat each image as its own patient


def load_gray(path, size=IMG_SIZE):
    """Read as grayscale, resize, apply CLAHE, scale to [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return img.astype(np.float32) / 255.0
