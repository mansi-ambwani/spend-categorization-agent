"""Clean the raw procurement export and split it into train / val / test.

Split strategy (to measure *generalisation*, not memorisation):
  * 15% of distinct catalogue items are held out completely -> all their rows go to test
    ("unseen items": products the model has never seen during training)
  * from the remaining rows: 10% -> test ("seen items"), 10% -> val, 80% -> train

Run:  python -m src.prepare_data
"""
import json
import re

import numpy as np
import pandas as pd

from src import config


def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace(" ", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("supplier_name", "line_description", "department"):
        df[col] = df[col].map(clean_text)
    df = df[df["line_description"].str.len() > 0]
    df = df.drop_duplicates(subset=["transaction_id"])
    return df.reset_index(drop=True)


def to_jsonl(df: pd.DataFrame, path, split_tag: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            f.write(json.dumps({
                "transaction_id": r.transaction_id,
                "supplier_name": r.supplier_name,
                "line_description": r.line_description,
                "department": r.department,
                "amount_inr": r.amount_inr,
                "category": r.category,
                "unseen_item": bool(getattr(r, "unseen_item", False)),
                "split": split_tag,
            }) + "\n")


def main():
    df = clean(pd.read_csv(config.RAW_DATA_PATH))
    rng = np.random.default_rng(config.SEED)

    # hold out ~15% of items in every category (stratified so each category keeps training data)
    held_out = set()
    for _, items in df.groupby("category")["_item"]:
        uniq = sorted(items.unique())
        k = max(1, round(0.15 * len(uniq)))
        held_out.update(rng.choice(uniq, size=k, replace=False))
    df["unseen_item"] = df["_item"].isin(held_out)

    unseen = df[df["unseen_item"]]
    seen = df[~df["unseen_item"]].sample(frac=1.0, random_state=config.SEED)
    n = len(seen)
    test_seen = seen.iloc[: int(0.10 * n)]
    val = seen.iloc[int(0.10 * n): int(0.20 * n)]
    train = seen.iloc[int(0.20 * n):]
    test = pd.concat([test_seen, unseen]).sample(frac=1.0, random_state=config.SEED)

    to_jsonl(train, config.TRAIN_PATH, "train")
    to_jsonl(val, config.VAL_PATH, "val")
    to_jsonl(test, config.TEST_PATH, "test")
    print(f"train={len(train)}  val={len(val)}  test={len(test)} "
          f"(of which unseen-item rows={len(unseen)}, held-out items={len(held_out)})")


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    main()
