"""End-to-end smoke tests (CPU only, ~1 minute).

Run:  python -m pytest -q
The LLM tests use a tiny random Llama model, so they check that the code runs,
not that predictions are good.
"""
import os

import pandas as pd
import pytest

from src import config
from src.agent.agent import SpendAgent, extract_sql
from src.agent.spend_db import SpendDatabase, format_inr
from src.prompts import category_names, snap_to_category


@pytest.fixture(scope="session")
def categorized():
    from src.categorize import categorize_df

    return categorize_df(pd.read_csv(config.RAW_DATA_PATH).head(400), "baseline")


def test_taxonomy_has_15_categories():
    assert len(category_names()) == 15


def test_baseline_predicts_valid_categories(categorized):
    assert set(categorized["predicted_category"]) <= set(category_names())
    assert categorized["confidence"].between(0, 1).all()


def test_sql_guard_blocks_writes(categorized):
    db = SpendDatabase(categorized)
    for bad in ["DELETE FROM spend", "DROP TABLE spend", "SELECT 1; DROP TABLE spend",
                "WITH x AS (SELECT 1) DELETE FROM spend"]:
        with pytest.raises(Exception):
            db.run_sql(bad)
    assert len(db.run_sql("SELECT * FROM spend LIMIT 3")) == 3


@pytest.mark.parametrize("q", ["total spend", "top 5 suppliers by spend", "spend by category",
                               "monthly trend of travel spend", "how much did we spend on software in 2025",
                               "which department spends the most on cloud", "average invoice value for logistics",
                               "how many transactions need review", "largest invoices", "spend in FY26"])
def test_rule_agent_answers(categorized, q):
    r = SpendAgent(SpendDatabase(categorized)).ask(q)
    assert r.sql and r.table is not None and r.answer


class FakeLLM:
    """Pretends to be an LLM: first returns broken SQL, then valid SQL, then an answer."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    def chat(self, system, user, temperature=0.0):
        self.calls += 1
        if "Query result" in user:
            return "Software & SaaS is the largest category."
        if "failed with error" in user:
            return "```sql\nSELECT category, SUM(amount_inr) AS s FROM spend GROUP BY category ORDER BY s DESC\n```"
        return "```sql\nSELECT categoryy FROM spend\n```"


def test_llm_agent_repairs_bad_sql(categorized):
    llm = FakeLLM()
    r = SpendAgent(SpendDatabase(categorized), llm).ask("biggest category?")
    assert r.backend == "fake" and llm.calls == 3 and len(r.table) > 1 and r.notes


def test_helpers():
    assert format_inr(12345678.4) == "Rs 1,23,45,678"
    assert extract_sql("blah ```sql\nSELECT 1\n``` blah") == "SELECT 1"
    cats = category_names()
    assert snap_to_category("software & saas.", cats) == "Software & SaaS"
    assert snap_to_category("Category: Raw Materials", cats) == "Raw Materials"


@pytest.mark.skipif(os.getenv("SKIP_LLM_TESTS") == "1", reason="LLM test disabled")
def test_tiny_llm_train_and_predict(tmp_path, monkeypatch):
    """Train 2 steps of LoRA on a tiny random Llama and run batched generation."""
    import sys

    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tests.make_tiny_model import main as make_tiny

    tiny = tmp_path / "tiny"
    make_tiny(tiny)
    adapter = tmp_path / "adapter"
    from src import train_qlora

    monkeypatch.setattr(sys, "argv", ["x", "--base-model", str(tiny), "--output-dir", str(adapter),
                                      "--max-train-samples", "16", "--max-steps", "2", "--batch-size", "4",
                                      "--grad-accum", "1"])
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    train_qlora.main()
    assert (adapter / "adapter_config.json").exists()

    from src.llm_classifier import LLMSpendClassifier

    clf = LLMSpendClassifier(adapter, str(tiny))
    preds = clf.predict([{"supplier_name": "Dell", "line_description": "laptop", "department": "IT"}] * 3,
                        batch_size=2, progress=False)
    assert len(preds) == 3 and all(p["category"] in category_names() for p in preds)
