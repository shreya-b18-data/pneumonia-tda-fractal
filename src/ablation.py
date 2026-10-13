"""Ablation: which feature groups carry the signal?
Usage:  python src/ablation.py      (writes results/ablation.json)"""
import json

import numpy as np
from sklearn.base import clone
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score

from config import RESULTS_DIR
from fractal import FRACTAL_NAMES
from topology import topological_feature_names
from train import MODELS, load

NAMES = topological_feature_names() + FRACTAL_NAMES


def groups():
    idx = lambda pred: [i for i, n in enumerate(NAMES) if pred(n)]
    return {
        "topology H0 only": idx(lambda n: "_H0_" in n),
        "topology H1 only": idx(lambda n: "_H1_" in n),
        "topology (H0+H1)": idx(lambda n: n.startswith(("sub_", "super_"))),
        "fractal only": idx(lambda n: n in FRACTAL_NAMES),
        "topology + fractal": list(range(len(NAMES))),
    }


def main():
    Xtr, ytr, gtr = load("train")
    Xte, yte, _ = load("test")
    assert Xtr.shape[1] == len(NAMES), f"expected {len(NAMES)} features, got {Xtr.shape[1]}"
    out = {}
    print(f"{'feature set':22s} {'n':>4s} {'CV AUC':>8s} {'test AUC':>9s} {'test recall':>12s}")
    for name, cols in groups().items():
        model = clone(MODELS["svm_rbf"])
        cv = cross_val_score(model, Xtr[:, cols], ytr, groups=gtr, cv=GroupKFold(5), scoring="roc_auc", n_jobs=-1)
        model.fit(Xtr[:, cols], ytr)
        prob = model.predict_proba(Xte[:, cols])[:, 1]
        out[name] = {"n_features": len(cols), "cv_auc": float(cv.mean()), "test_auc": float(roc_auc_score(yte, prob)),
                     "test_recall": float(recall_score(yte, prob >= 0.5))}
        print(f"{name:22s} {len(cols):4d} {cv.mean():8.3f} {out[name]['test_auc']:9.3f} {out[name]['test_recall']:12.3f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ablation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
