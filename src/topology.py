"""Persistent homology features from cubical complexes (sublevel and superlevel filtrations)."""
import gudhi
import numpy as np

N_BETTI = 16
_STAT_NAMES = ["n", "total", "mean", "std", "max", "entropy"]


def _intervals(img):
    """Persistence intervals of the sublevel-set filtration, split by homology dimension.
    Essential classes (death = inf) are clipped to 1.0, the maximum pixel value."""
    cc = gudhi.CubicalComplex(top_dimensional_cells=img.astype(np.float64))
    cc.compute_persistence()
    out = {0: [], 1: []}
    for dim, (b, d) in cc.persistence():
        if dim in out:
            out[dim].append((b, 1.0 if np.isinf(d) else d))
    return {k: np.array(v).reshape(-1, 2) for k, v in out.items()}


def _summary(iv):
    """Summary statistics + Betti curve for one set of (birth, death) intervals."""
    if len(iv) == 0:
        return [0.0] * (len(_STAT_NAMES) + N_BETTI)
    life = iv[:, 1] - iv[:, 0]
    p = life / life.sum() if life.sum() > 0 else np.ones_like(life) / len(life)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    stats = [len(life), life.sum(), life.mean(), life.std(), life.max(), entropy]
    ts = np.linspace(0, 1, N_BETTI)
    betti = [float(((iv[:, 0] <= t) & (t < iv[:, 1])).sum()) for t in ts]
    return stats + betti


def topological_features(img):
    """img in [0,1], 2D. Returns a fixed-length vector: 2 filtrations x 2 dimensions x (6 stats + 16 Betti values)."""
    feats = []
    for f in (img, 1.0 - img):          # sublevel of image, then of its inverse (= superlevel sets)
        iv = _intervals(f)
        for dim in (0, 1):
            feats += _summary(iv[dim])
    return np.array(feats, dtype=np.float32)


def topological_feature_names():
    names = []
    for filt in ("sub", "super"):
        for dim in (0, 1):
            names += [f"{filt}_H{dim}_{s}" for s in _STAT_NAMES]
            names += [f"{filt}_H{dim}_betti{i}" for i in range(N_BETTI)]
    return names
