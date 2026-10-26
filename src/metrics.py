"""Shared evaluation helpers: accuracy, macro-F1, per-class report, confusion matrix plot."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, unseen_mask=None, labels=None) -> dict:
    out = {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0), 4),
    }
    if unseen_mask is not None:
        m = np.asarray(unseen_mask, dtype=bool)
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if m.any():
            out["accuracy_unseen_items"] = round(accuracy_score(yt[m], yp[m]), 4)
            out["n_unseen_items"] = int(m.sum())
        if (~m).any():
            out["accuracy_seen_items"] = round(accuracy_score(yt[~m], yp[~m]), 4)
            out["n_seen_items"] = int((~m).sum())
    out["per_class"] = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return out


def save_metrics(metrics: dict, path: Path):
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion(y_true, y_pred, labels, path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_summary(name: str, m: dict):
    line = f"[{name}] accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}"
    if "accuracy_unseen_items" in m:
        line += f"  seen={m.get('accuracy_seen_items', float('nan')):.4f}  unseen={m['accuracy_unseen_items']:.4f}"
    print(line)
