"""Classical baseline: TF-IDF (word + character n-grams) + Logistic Regression.

Runs on any CPU in a few seconds.  Used (1) as the comparison point for the
fine-tuned LLM and (2) as a fast fallback classifier when no GPU is available.

Run:  python -m src.baseline
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config
from src.metrics import compute_metrics, plot_confusion, print_summary, save_metrics
from src.prepare_data import load_jsonl
from src.prompts import category_names, record_text


def _texts(rows):
    return [record_text(r["supplier_name"], r["line_description"], r.get("department", "")) for r in rows]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


class BaselineClassifier:
    """Thin wrapper with the same predict() interface as the LLM classifier."""

    def __init__(self, path=config.BASELINE_MODEL_PATH):
        self.pipe = joblib.load(path)

    def predict(self, records: list[dict], batch_size: int = 0):
        probs = self.pipe.predict_proba(_texts(records))
        classes = self.pipe.classes_
        idx = probs.argmax(axis=1)
        return [{"category": classes[i], "confidence": float(p[i])} for i, p in zip(idx, probs)]


def main():
    train, val, test = (load_jsonl(p) for p in (config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH))
    pipe = build_pipeline()
    pipe.fit(_texts(train + val), [r["category"] for r in train + val])
    joblib.dump(pipe, config.BASELINE_MODEL_PATH)

    labels = category_names()
    y_true = [r["category"] for r in test]
    y_pred = list(pipe.predict(_texts(test)))
    m = compute_metrics(y_true, y_pred, [r["unseen_item"] for r in test], labels)
    m["model"] = "TF-IDF + Logistic Regression"
    save_metrics(m, config.REPORTS_DIR / "metrics_baseline.json")
    plot_confusion(y_true, y_pred, labels, config.REPORTS_DIR / "confusion_baseline.png",
                   f"Baseline TF-IDF + LogReg (acc={m['accuracy']:.3f})")
    print_summary("baseline", m)
    print(f"Saved model -> {config.BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()
