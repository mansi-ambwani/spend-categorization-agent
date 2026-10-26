"""Loads categorized spend into an in-memory SQLite database and runs *safe* read-only SQL."""
import re
import sqlite3

import pandas as pd

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b",
                       re.IGNORECASE)
MAX_ROWS = 500


def format_inr(x: float) -> str:
    """Indian digit grouping: 12345678.9 -> 'Rs 1,23,45,679'."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    neg, x = x < 0, abs(round(x))
    s = str(int(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"{'-' if neg else ''}Rs {s}"


def format_crore_lakh(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e7:
        return f"Rs {x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x / 1e5:,.2f} L"
    return format_inr(x)


class SpendDatabase:
    """Table `spend` with one row per invoice line."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        if "predicted_category" in df.columns:
            df["category"] = df["predicted_category"]
        elif "category" not in df.columns:
            raise ValueError("Data must have a 'category' or 'predicted_category' column - categorize it first.")
        df = df.drop(columns=[c for c in ("predicted_category", "true_category", "classifier", "raw_output")
                              if c in df.columns])
        df = df.rename(columns={"line_description": "description", "supplier_name": "supplier"})
        df["invoice_date"] = pd.to_datetime(df.get("invoice_date"), errors="coerce")
        df["month"] = df["invoice_date"].dt.strftime("%Y-%m")
        df["quarter"] = df["invoice_date"].dt.year.astype("Int64").astype(str) + "-Q" + \
            df["invoice_date"].dt.quarter.astype("Int64").astype(str)
        df["invoice_date"] = df["invoice_date"].dt.strftime("%Y-%m-%d")
        if "amount_inr" not in df.columns:
            df["amount_inr"] = 0.0
        if "needs_review" in df.columns:
            df["needs_review"] = df["needs_review"].astype(int)
        keep = [c for c in ("transaction_id", "invoice_date", "month", "quarter", "supplier", "description",
                            "department", "category", "amount_inr", "confidence", "needs_review") if c in df.columns]
        self.df = df[keep]
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.df.to_sql("spend", self.conn, index=False)
        self.conn.execute("PRAGMA query_only = ON")  # database becomes read-only for every later query

    @classmethod
    def from_csv(cls, path) -> "SpendDatabase":
        return cls(pd.read_csv(path))

    # ------------------------------------------------------------------ metadata for prompts
    @property
    def categories(self) -> list[str]:
        return sorted(self.df["category"].dropna().unique().tolist())

    @property
    def suppliers(self) -> list[str]:
        return sorted(self.df["supplier"].dropna().unique().tolist())

    @property
    def departments(self) -> list[str]:
        return sorted(self.df["department"].dropna().unique().tolist()) if "department" in self.df else []

    @property
    def date_range(self) -> tuple[str, str]:
        d = self.df["invoice_date"].dropna()
        return (d.min(), d.max()) if len(d) else ("", "")

    def schema_description(self) -> str:
        cols = {
            "transaction_id": "TEXT unique id of the invoice line",
            "invoice_date": "TEXT date 'YYYY-MM-DD'",
            "month": "TEXT 'YYYY-MM'",
            "quarter": "TEXT calendar quarter like '2025-Q3'",
            "supplier": "TEXT supplier / vendor name",
            "description": "TEXT free-text line description",
            "department": "TEXT department that raised the purchase",
            "category": "TEXT spend category (predicted by the fine-tuned model)",
            "amount_inr": "REAL invoice line amount in Indian Rupees",
            "confidence": "REAL classifier confidence 0-1",
            "needs_review": "INTEGER 1 if the category prediction is low-confidence",
        }
        lines = [f"  {c} {cols[c]}" for c in self.df.columns if c in cols]
        lo, hi = self.date_range
        return (
            "SQLite table `spend`:\n" + "\n".join(lines) +
            f"\nRows: {len(self.df)}. Dates from {lo} to {hi}.\n"
            f"Categories: {', '.join(self.categories)}\n"
            f"Departments: {', '.join(self.departments)}"
        )

    # ------------------------------------------------------------------ query
    def run_sql(self, sql: str) -> pd.DataFrame:
        sql = sql.strip().rstrip(";").strip()
        if ";" in sql:
            raise ValueError("Only a single SQL statement is allowed.")
        if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        if FORBIDDEN.search(sql):
            raise ValueError("Query contains a forbidden keyword.")
        return pd.read_sql_query(sql, self.conn).head(MAX_ROWS)
