"""Prompt used for both fine-tuning and inference (they must match exactly)."""
import difflib
import json
from functools import lru_cache

from src import config

SYSTEM_PROMPT = (
    "You are a procurement spend-classification expert. Read one purchase line item "
    "(supplier, free-text description, department) and assign it to exactly one spend "
    "category from the list. Reply with the category name only."
)


@lru_cache(maxsize=1)
def load_taxonomy() -> list[dict]:
    return json.loads(config.TAXONOMY_PATH.read_text(encoding="utf-8"))


def category_names() -> list[str]:
    return [t["category"] for t in load_taxonomy()]


def build_user_prompt(supplier: str, description: str, department: str = "") -> str:
    # category names only: keeps prompts short (~180 tokens) so training on a free T4 GPU is fast.
    # The full category definitions live in data/taxonomy.json.
    cats = "\n".join(f"- {c}" for c in category_names())
    description = str(description)[:300]
    return (
        f"Spend categories:\n{cats}\n\n"
        f"Supplier: {supplier}\n"
        f"Description: {description}\n"
        f"Department: {department or 'unknown'}\n\n"
        "Category:"
    )


def build_messages(supplier: str, description: str, department: str = "") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(supplier, description, department)},
    ]


def record_text(supplier: str, description: str, department: str = "") -> str:
    """Flat text used by the TF-IDF baseline."""
    return f"{supplier} | {description} | {department}"


def snap_to_category(text: str, categories: list[str]) -> str:
    t = text.strip().strip(".").strip()
    for c in categories:
        if t.lower() == c.lower():
            return c
    for c in categories:  # model said a bit more than the label
        if c.lower() in t.lower():
            return c
    match = difflib.get_close_matches(t, categories, n=1, cutoff=0.0)
    return match[0] if match else categories[0]
