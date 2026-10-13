"""ROC curve and confusion matrices for the selected model.
Usage:  python src/plots.py      (writes results/figures/*.png)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from config import CLASSES, RESULTS_DIR
from train import MODELS, load


def main():
    Xtr, ytr, _ = load("train")
    Xte, yte, _ = load("test")
    prob = clone(MODELS["svm_rbf"]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(yte, prob)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc_score(yte, prob):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve, test split")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(fig_dir / "roc_curve.png", dpi=150); plt.close()

    cm = confusion_matrix(yte, prob >= 0.5)
    plt.figure(figsize=(4, 3.6))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], CLASSES); plt.yticks([0, 1], CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix, threshold 0.5")
    plt.tight_layout(); plt.savefig(fig_dir / "confusion_matrix.png", dpi=150); plt.close()
    print("saved", [p.name for p in fig_dir.glob("*.png")])


if __name__ == "__main__":
    main()
