"""Categorize a procurement CSV (new / uncategorized spend) and write an enriched CSV.

Input CSV needs at least: supplier_name, line_description
Optional columns: department, amount_inr, invoice_date, transaction_id (kept as-is)

Examples:
  python -m src.categorize --model llm                    # fine-tuned Llama-3 on the sample data
  python -m src.categorize --model baseline --input my_spend.csv --output my_spend_categorized.csv
"""
import argparse

import pandas as pd

from src import config

REQUIRED = ["supplier_name", "line_description"]


def categorize_df(df: pd.DataFrame, model: str = "baseline", classifier=None, batch_size: int = 16,
                  threshold: float = config.REVIEW_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")
    df = df.copy()
    if "department" not in df.columns:
        df["department"] = ""
    df = df.fillna({"supplier_name": "", "line_description": "", "department": ""})
    records = df[["supplier_name", "line_description", "department"]].astype(str).to_dict("records")
    if classifier is None:
        if model == "llm":
            from src.llm_classifier import LLMSpendClassifier
            classifier = LLMSpendClassifier()
        else:
            from src.baseline import BaselineClassifier
            classifier = BaselineClassifier()
    preds = classifier.predict(records, batch_size=batch_size)
    if "category" in df.columns:  # keep ground truth (if any) under a different name
        df = df.rename(columns={"category": "true_category"})
    df["predicted_category"] = [p["category"] for p in preds]
    df["confidence"] = [round(p["confidence"], 4) for p in preds]
    df["needs_review"] = df["confidence"] < threshold
    df["classifier"] = "llama3-8b-qlora" if model == "llm" else "tfidf-logreg"
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["llm", "baseline"], default="llm")
    ap.add_argument("--input", default=str(config.RAW_DATA_PATH))
    ap.add_argument("--output", default=str(config.CATEGORIZED_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    out = categorize_df(df, args.model, batch_size=args.batch_size)
    out.to_csv(args.output, index=False)
    print(f"Categorized {len(out)} rows -> {args.output}")
    print(f"Flagged for human review (confidence < {config.REVIEW_CONFIDENCE_THRESHOLD}): "
          f"{int(out['needs_review'].sum())}")
    if "true_category" in out.columns:
        acc = (out["true_category"] == out["predicted_category"]).mean()
        print(f"Agreement with ground-truth labels: {acc:.3f}")
    print(out.groupby("predicted_category")["amount_inr"].sum().sort_values(ascending=False)
          .map(lambda x: f"Rs {x:,.0f}").to_string() if "amount_inr" in out.columns else "")


if __name__ == "__main__":
    main()
