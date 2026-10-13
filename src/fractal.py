"""Fractal features: box-counting dimension of binarised / edge images and lacunarity."""
import cv2
import numpy as np

_SIZES = (2, 4, 8, 16, 32)


def box_count_dimension(binary):
    """Box-counting dimension of a 2D boolean image (slope of log N(s) vs log(1/s))."""
    h, w = binary.shape
    counts = []
    for s in _SIZES:
        hh, ww = (h // s) * s, (w // s) * s
        blocks = binary[:hh, :ww].reshape(hh // s, s, ww // s, s)
        counts.append(max(1, int(blocks.any(axis=(1, 3)).sum())))
    slope = np.polyfit(np.log(1.0 / np.array(_SIZES)), np.log(counts), 1)[0]
    return float(slope)


def lacunarity(gray, box):
    """Lacunarity with non-overlapping boxes: 1 + var(mass) / mean(mass)^2."""
    h, w = gray.shape
    hh, ww = (h // box) * box, (w // box) * box
    mass = gray[:hh, :ww].reshape(hh // box, box, ww // box, box).sum(axis=(1, 3)).ravel()
    m = mass.mean()
    return float(1.0 + mass.var() / (m ** 2 + 1e-12))


def fractal_features(img):
    """img in [0,1]. Returns [D_otsu, D_canny, D_mean, lacunarity_8, lacunarity_16]."""
    u8 = (img * 255).astype(np.uint8)
    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(u8, 50, 150)
    return np.array([
        box_count_dimension(otsu > 0),
        box_count_dimension(edges > 0),
        box_count_dimension(img > img.mean()),
        lacunarity(img, 8),
        lacunarity(img, 16),
    ], dtype=np.float32)


FRACTAL_NAMES = ["fd_otsu", "fd_canny", "fd_mean", "lacunarity_8", "lacunarity_16"]
