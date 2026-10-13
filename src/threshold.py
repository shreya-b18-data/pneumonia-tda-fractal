"""Choose the decision threshold from out-of-fold predictions on TRAIN only, then apply it once to test.
Usage:  python src/threshold.py      (writes results/threshold_tuning.json)"""
import json

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from config import RESULTS_DIR
from train import MODELS, load


def metrics_at(prob, y, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": float(thr), "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2, "f1": float(f1_score(y, pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def youden_threshold(prob, y):
    """Threshold maximising recall + specificity - 1 (Youden's J)."""
    ts = np.linspace(0.05, 0.95, 91)
    j = [metrics_at(prob, y, t)["recall"] + metrics_at(prob, y, t)["specificity"] - 1 for t in ts]
    return float(ts[int(np.argmax(j))])


def main():
    Xtr, ytr, gtr = load("train")
    Xte, yte, _ = load("test")
    model = clone(MODELS["svm_rbf"])
    oof = cross_val_predict(model, Xtr, ytr, groups=gtr, cv=GroupKFold(5), method="predict_proba", n_jobs=-1)[:, 1]
    thr = youden_threshold(oof, ytr)
    prob = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    out = {"test_roc_auc": float(roc_auc_score(yte, prob)), "threshold_chosen_on": "train out-of-fold predictions",
           "default_0.5": metrics_at(prob, yte, 0.5), "tuned": metrics_at(prob, yte, thr)}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "threshold_tuning.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
