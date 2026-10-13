"""Extract topological + fractal features for every image.
Usage:  python src/extract.py            (writes features/{train,val,test}.npz)"""
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from config import FEATURE_DIR
from dataset import index_split, load_gray, patient_id
from fractal import fractal_features
from topology import topological_features


def image_features(path):
    img = load_gray(path)
    return np.concatenate([topological_features(img), fractal_features(img)])


def main():
    FEATURE_DIR.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        paths, y = index_split(split)
        if not paths:
            print(f"[skip] no images found for split '{split}'")
            continue
        X = np.stack(Parallel(n_jobs=-1)(delayed(image_features)(p) for p in tqdm(paths, desc=split)))
        groups = np.array([patient_id(p) for p in paths])
        np.savez(FEATURE_DIR / f"{split}.npz", X=X, y=y, groups=groups, paths=np.array([str(p) for p in paths]))
        print(f"{split}: X={X.shape}, pneumonia={int(y.sum())}, normal={int((y == 0).sum())}")


if __name__ == "__main__":
    main()
