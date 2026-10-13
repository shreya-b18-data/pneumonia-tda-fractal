"""Train and evaluate classifiers on the extracted features.
Model selection: patient-grouped 5-fold CV on train. Final numbers: the official test split, evaluated once.
Usage:  python src/train.py"""
import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import FEATURE_DIR, RESULTS_DIR, SEED

MODELS = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)),
    "random_forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "grad_boost": GradientBoostingClassifier(random_state=SEED),
}


def load(split):
    d = np.load(FEATURE_DIR / f"{split}.npz", allow_pickle=True)
    return d["X"], d["y"], d["groups"]


def report(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "specificity": tn / (tn + fp), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def main():
    Xtr, ytr, gtr = load("train")
    Xte, yte, _ = load("test")
    RESULTS_DIR.mkdir(exist_ok=True)

    cv = GroupKFold(n_splits=5)
    cv_scores = {}
    for name, m in MODELS.items():
        s = cross_val_score(m, Xtr, ytr, groups=gtr, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = {"cv_auc_mean": float(s.mean()), "cv_auc_std": float(s.std())}
        print(f"{name:14s} CV AUC {s.mean():.3f} +/- {s.std():.3f}")

    best = max(cv_scores, key=lambda k: cv_scores[k]["cv_auc_mean"])
    print(f"\nBest by CV: {best}. Evaluating once on the official test split.")
    model = MODELS[best].fit(Xtr, ytr)
    test = report(model, Xte, yte)
    out = {"cv": cv_scores, "selected_model": best, "test": test}
    (RESULTS_DIR / "tda_fractal_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
