"""Evaluate a classifier on the held-out test set.

Examples:
  python -m src.evaluate --model llm                        # fine-tuned Llama-3 (needs GPU)
  python -m src.evaluate --model llm --no-adapter --limit 300   # zero-shot Llama-3 (before fine-tuning)
  python -m src.evaluate --model baseline                   # TF-IDF baseline (CPU)
Writes reports/metrics_<name>.json, reports/confusion_<name>.png and reports/predictions_<name>.csv
"""
import argparse
import time

import pandas as pd

from src import config
from src.metrics import compute_metrics, plot_confusion, print_summary, save_metrics
from src.prepare_data import load_jsonl
from src.prompts import category_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["llm", "baseline"], default="llm")
    ap.add_argument("--adapter-dir", default=str(config.ADAPTER_DIR))
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--no-adapter", action="store_true", help="zero-shot base model")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on the first N test rows only")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--name", default=None, help="name used for the report files")
    args = ap.parse_args()

    test = load_jsonl(config.TEST_PATH)
    if args.limit:
        test = test[: args.limit]

    if args.model == "baseline":
        from src.baseline import BaselineClassifier

        clf, name, title = BaselineClassifier(), args.name or "baseline", "TF-IDF + Logistic Regression"
    else:
        from src.llm_classifier import LLMSpendClassifier

        clf = LLMSpendClassifier(None if args.no_adapter else args.adapter_dir, args.base_model,
                                 load_in_4bit=not args.no_4bit)
        name = args.name or ("llm_zero_shot" if args.no_adapter else "llm_qlora")
        title = "Llama-3 8B zero-shot" if args.no_adapter else "Llama-3 8B + QLoRA fine-tuned"

    t0 = time.time()
    preds = clf.predict(test, batch_size=args.batch_size)
    secs = time.time() - t0

    labels = category_names()
    y_true = [r["category"] for r in test]
    y_pred = [p["category"] for p in preds]
    m = compute_metrics(y_true, y_pred, [r["unseen_item"] for r in test], labels)
    m["model"] = title
    m["seconds"] = round(secs, 1)
    m["rows_per_second"] = round(len(test) / max(secs, 1e-9), 2)
    save_metrics(m, config.REPORTS_DIR / f"metrics_{name}.json")
    plot_confusion(y_true, y_pred, labels, config.REPORTS_DIR / f"confusion_{name}.png",
                   f"{title} (acc={m['accuracy']:.3f})")
    df = pd.DataFrame(test)
    df["predicted"] = y_pred
    df["confidence"] = [p["confidence"] for p in preds]
    df["correct"] = df["predicted"] == df["category"]
    df.to_csv(config.REPORTS_DIR / f"predictions_{name}.csv", index=False)
    print_summary(name, m)
    print("Most common mistakes:")
    errs = df[~df["correct"]].groupby(["category", "predicted"]).size().sort_values(ascending=False).head(8)
    print(errs.to_string())


if __name__ == "__main__":
    main()
