"""Natural-language spend agent.

Flow for one question (LLM mode):
  1. PLAN   - the LLM sees the table schema, category list, a few examples and recent chat
              history, and writes one read-only SQLite query (text-to-SQL tool call).
  2. ACT    - the query is validated (SELECT only, read-only connection) and executed.
  3. REPAIR - if SQLite raises an error, the error is sent back to the LLM once to fix the SQL.
  4. ANSWER - the LLM turns the result table into a short business answer (numbers in INR).
If no LLM is configured, or the LLM call fails, the offline rule-based parser answers instead,
so the agent always responds.

CLI:  python -m src.agent.agent "top 5 suppliers by spend"
      python -m src.agent.agent            (interactive chat)
"""
import re
import sys
from dataclasses import dataclass, field

import pandas as pd

from src import config
from src.agent.llm_backends import LLMError, get_backend
from src.agent.rules import RuleBasedParser, describe_result
from src.agent.spend_db import SpendDatabase

SQL_SYSTEM = """You are a procurement spend-analytics agent. You answer business questions by writing ONE SQLite query
against the table described below. Rules:
- Output only the SQL inside a ```sql code block. No explanation.
- Read-only: a single SELECT (CTEs allowed). Never modify data.
- Use the exact category names listed. Match suppliers with LIKE '%name%' (case-insensitive is fine).
- Money is amount_inr (Indian Rupees). Round aggregates with ROUND(x, 2). Give aggregates clear aliases.
- For rankings use ORDER BY ... DESC LIMIT n. For trends group by month (or quarter) and order by it.
- Indian financial year FY26 = 2025-04-01 to 2026-03-31.
- If the question cannot be answered from this table, output: NO_SQL: <short reply to the user>"""

FEW_SHOT = """Examples:
Q: total spend on software last quarter
```sql
SELECT ROUND(SUM(amount_inr), 2) AS total_spend_inr FROM spend
WHERE category = 'Software & SaaS' AND quarter = (SELECT MAX(quarter) FROM spend)
```
Q: top 3 vendors for travel
```sql
SELECT supplier, ROUND(SUM(amount_inr), 2) AS total_spend_inr, COUNT(*) AS transactions FROM spend
WHERE category = 'Travel & Entertainment' GROUP BY supplier ORDER BY total_spend_inr DESC LIMIT 3
```
Q: monthly marketing spend trend
```sql
SELECT month, ROUND(SUM(amount_inr), 2) AS total_spend_inr FROM spend
WHERE category = 'Marketing & Advertising' GROUP BY month ORDER BY month
```"""

ANSWER_SYSTEM = """You are a procurement spend-analytics assistant talking to a business user.
Using ONLY the query result provided, answer the question in 1-4 short sentences or a short list.
Format money in Indian Rupees with Indian digit grouping (e.g. Rs 12,34,567) and mention lakh/crore for big values.
Do not invent numbers that are not in the result. Do not mention SQL."""


@dataclass
class AgentResponse:
    answer: str
    sql: str | None = None
    table: pd.DataFrame | None = None
    backend: str = "rules"
    notes: list[str] = field(default_factory=list)


def extract_sql(text: str) -> str | None:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(SELECT|WITH)\b.*", text, re.DOTALL | re.IGNORECASE)
    return m.group(0).strip() if m else None


class SpendAgent:
    def __init__(self, db: SpendDatabase, backend=None, history_turns: int = 3):
        self.db = db
        self.backend = backend
        self.rules = RuleBasedParser(db)
        self.history: list[tuple[str, str]] = []
        self.history_turns = history_turns

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> AgentResponse:
        question = question.strip()
        if not question:
            return AgentResponse("Please ask a question about your spend data.")
        if self.backend is not None:
            try:
                resp = self._ask_llm(question)
            except (LLMError, KeyError, ValueError) as e:
                resp = self._ask_rules(question)
                resp.notes.append(f"LLM backend failed ({e}); answered with the offline parser instead.")
        else:
            resp = self._ask_rules(question)
        self.history.append((question, resp.answer))
        self.history = self.history[-self.history_turns:]
        return resp

    # ------------------------------------------------------------------ offline
    def _ask_rules(self, question: str) -> AgentResponse:
        if re.fullmatch(r"(hi|hello|hey|help|what can you do)\W*", question.lower()):
            return AgentResponse(self.help_text())
        pq = self.rules.parse(question)
        df = self.db.run_sql(pq.sql)
        return AgentResponse(describe_result(pq, df), pq.sql, df, "rules")

    # ------------------------------------------------------------------ LLM
    def _ask_llm(self, question: str) -> AgentResponse:
        hist = "\n".join(f"Previous Q: {q}\nPrevious A: {a[:300]}" for q, a in self.history)
        user = (f"{self.db.schema_description()}\n\n{FEW_SHOT}\n\n"
                f"{('Conversation so far:' + chr(10) + hist + chr(10) + chr(10)) if hist else ''}"
                f"Q: {question}")
        raw = self.backend.chat(SQL_SYSTEM, user)
        if "NO_SQL" in raw and not extract_sql(raw.split("NO_SQL")[0]):
            return AgentResponse(raw.split("NO_SQL:", 1)[-1].strip() or self.help_text(), backend=self.backend.name)
        sql = extract_sql(raw)
        if not sql:
            raise ValueError("LLM did not return SQL")
        notes = []
        try:
            df = self.db.run_sql(sql)
        except Exception as e:  # one repair attempt
            notes.append(f"First query failed ({e}); asked the LLM to fix it.")
            raw = self.backend.chat(SQL_SYSTEM, f"{user}\n\nYour previous query:\n```sql\n{sql}\n```\n"
                                                f"failed with error: {e}\nReturn a corrected query.")
            sql = extract_sql(raw)
            if not sql:
                raise ValueError("LLM did not return SQL on repair")
            df = self.db.run_sql(sql)
        result_txt = df.head(30).to_string(index=False) if not df.empty else "(no rows)"
        answer = self.backend.chat(ANSWER_SYSTEM, f"Question: {question}\n\nQuery result:\n{result_txt}")
        return AgentResponse(answer.strip(), sql, df, self.backend.name, notes)

    def help_text(self) -> str:
        lo, hi = self.db.date_range
        return ("I can answer questions about your categorized spend "
                f"({len(self.db.df)} invoice lines, {lo} to {hi}). Try:\n"
                "- What is our total spend?\n- Top 5 suppliers by spend\n- Spend by category\n"
                "- How much did we spend on software in 2025?\n- Monthly trend of travel spend\n"
                "- Which department spends the most on cloud?\n- How many transactions need review?")


def main():
    path = config.CATEGORIZED_PATH if config.CATEGORIZED_PATH.exists() else config.RAW_DATA_PATH
    db = SpendDatabase.from_csv(path)
    try:
        backend = get_backend()
    except LLMError as e:
        print(f"[{e}] -> using offline rule-based mode")
        backend = None
    agent = SpendAgent(db, backend)
    print(f"Loaded {len(db.df)} rows from {path.name}. Backend: {backend.name if backend else 'rules'}")
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    while True:
        q = questions.pop() if questions else input("\nAsk about spend (or 'exit'): ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        r = agent.ask(q)
        print("\n" + r.answer)
        if r.sql:
            print(f"\n[SQL] {r.sql}")
        for n in r.notes:
            print(f"[note] {n}")
        if questions is not None:
            break


if __name__ == "__main__":
    main()
