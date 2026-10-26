"""Offline question parser: turns common spend questions into SQL without any LLM / API key.

Handles questions such as
  "total spend"                         "top 5 suppliers by spend"
  "how much did we spend on software?"  "monthly trend of travel spend"
  "spend by department in 2025"         "which category has the highest spend in Q4?"
  "how many transactions need review?"  "average invoice value for logistics"
It is the default backend and the safety net when the LLM backend is not reachable.
"""
import re
from dataclasses import dataclass

from src.agent.spend_db import SpendDatabase

CATEGORY_SYNONYMS = {
    "IT Hardware": ["hardware", "laptop", "laptops", "monitor", "server", "servers", "printer", "computer",
                    "computers", "devices", "it equipment"],
    "Software & SaaS": ["software", "saas", "licence", "licences", "license", "licenses", "subscription",
                        "subscriptions", "apps"],
    "Cloud & Telecom Services": ["cloud", "telecom", "internet", "broadband", "hosting",
                                 "it services", "network services", "telephony"],
    "Office Supplies": ["office supplies", "stationery", "stationary", "toner", "paper", "pens"],
    "Facilities & Maintenance": ["facilities", "facility", "maintenance", "housekeeping", "security guard",
                                 "rent", "furniture", "repairs"],
    "Professional Services": ["professional services", "consulting", "consultancy", "legal", "audit", "lawyers",
                              "advisory", "tax advisory"],
    "Marketing & Advertising": ["marketing", "advertising", "ads", "advertisement", "campaign", "campaigns",
                                "branding", "promotion"],
    "Travel & Entertainment": ["travel", "flights", "flight", "hotel", "hotels", "cab", "cabs",
                               "entertainment", "t&e"],
    "Logistics & Freight": ["logistics", "freight", "shipping", "courier", "transport", "warehousing",
                            "shipment", "shipments"],
    "Raw Materials": ["raw material", "raw materials", "steel", "aluminium", "polymer", "resin",
                      "direct material", "direct materials"],
    "MRO & Industrial Supplies": ["mro", "industrial supplies", "industrial", "safety", "ppe", "tools",
                                  "spares", "spare parts", "lubricants"],
    "Utilities & Energy": ["utilities", "utility", "electricity", "energy", "power", "water", "fuel", "diesel",
                           "gas"],
    "HR & Training": ["hr", "training", "recruitment", "hiring", "staffing", "learning", "insurance",
                      "people"],
    "Food & Catering": ["food", "catering", "cafeteria", "pantry", "meals", "snacks", "canteen"],
    "Packaging Materials": ["packaging", "packing", "boxes", "cartons", "pallets"],
}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
GENERIC_SUPPLIER_WORDS = {"india", "ltd", "pvt", "private", "limited", "the", "and", "company", "co", "inc",
                          "corporation", "services", "group", "international", "llp", "local", "global",
                          "solutions", "systems", "technologies", "industries", "misc", "vendor", "supplier",
                          "enterprises", "traders", "business", "express", "retail", "a", "of"}

DIMENSIONS = {
    "category": ("category", "category"),
    "supplier": ("supplier", "supplier"),
    "department": ("department", "department"),
    "month": ("month", "month"),
    "quarter": ("quarter", "quarter"),
}


@dataclass
class ParsedQuery:
    sql: str
    kind: str          # "scalar" | "grouped" | "trend" | "rows"
    metric: str        # "sum" | "count" | "avg" | ...
    dimension: str | None
    filters: list[str]
    labels: list[str] | None = None  # human-readable version of the filters


def _word_in(word: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) is not None


def _esc(s: str) -> str:
    return s.replace("'", "''")


class RuleBasedParser:
    def __init__(self, db: SpendDatabase):
        self.db = db
        # distinctive supplier keywords, e.g. "Dell Technologies India Pvt Ltd" -> "dell"
        self.supplier_keys: dict[str, list[str]] = {}
        for s in db.suppliers:
            low = s.lower()
            self.supplier_keys.setdefault(low, []).append(s)
            toks = [t for t in re.findall(r"[a-z0-9&]+", low) if t not in GENERIC_SUPPLIER_WORDS and len(t) >= 3]
            if toks:
                self.supplier_keys.setdefault(toks[0], []).append(s)
        self.supplier_keys.setdefault("aws", []).extend([s for s in db.suppliers if "amazon web" in s.lower()])

    # ------------------------------------------------------------------ helpers
    def _categories(self, q: str) -> list[str]:
        found = []
        for cat in self.db.categories:
            names = [cat.lower()] + CATEGORY_SYNONYMS.get(cat, [])
            if any(_word_in(n, q) for n in names):
                found.append(cat)
        return found

    def _suppliers(self, q: str) -> list[str]:
        found = []
        for key, sups in self.supplier_keys.items():
            if _word_in(key, q):
                found.extend(sups)
        return sorted(set(found))

    def _departments(self, q: str) -> list[str]:
        out = []
        for d in self.db.departments:
            dl = d.lower()
            if _word_in(dl, q) and not (dl in ("hr", "it") and f"{dl} &" in q):
                out.append(d)
        return out

    def _time_filters(self, q: str) -> list[str]:
        f = []
        years = re.findall(r"\b(20\d\d)\b", q)
        fy = re.search(r"\bfy\s?'?(20)?(\d\d)(?:-(\d\d))?\b", q)
        if fy:  # Indian financial year FY26 = Apr 2025 - Mar 2026
            end = 2000 + int(fy.group(2))
            f.append(f"invoice_date BETWEEN '{end - 1}-04-01' AND '{end}-03-31'")
            years = []
        month_hits = [m for m in MONTHS if _word_in(m, q)]
        quarter = re.search(r"\bq([1-4])\b", q)
        if month_hits:
            m = MONTHS[month_hits[0]]
            if years:
                f.append(f"month = '{years[0]}-{m:02d}'")
            else:
                f.append(f"CAST(substr(month, 6, 2) AS INTEGER) = {m}")
        elif quarter:
            qn = quarter.group(1)
            f.append(f"quarter = '{years[0]}-Q{qn}'" if years else f"substr(quarter, 6) = 'Q{qn}'")
        elif years:
            f.append(f"substr(invoice_date, 1, 4) = '{years[0]}'")
        last = re.search(r"\blast (\d+) months?\b", q) or (re.search(r"\blast month\b", q) and "1")
        if last:
            n = int(last if isinstance(last, str) else last.group(1))
            f.append(f"invoice_date > date((SELECT MAX(invoice_date) FROM spend), '-{n} months')")
        return f

    def _time_label(self, q: str) -> str:
        parts = []
        for pat in (r"\bfy\s?'?\d{2,4}(?:-\d\d)?\b", r"\bq[1-4]\b", r"\b20\d\d\b", r"\blast \d* ?months?\b"):
            parts += re.findall(pat, q)
        parts += [m for m in MONTHS if len(m) > 3 and _word_in(m, q)] or [m for m in MONTHS if len(m) == 3 and _word_in(m, q)]
        return " ".join(p.upper() if p.startswith(("fy", "q")) else p for p in parts) or "selected period"

    # ------------------------------------------------------------------ main entry
    def parse(self, question: str) -> ParsedQuery:
        q = " " + question.lower().strip().rstrip("?") + " "
        filters = []
        cats, sups, depts = self._categories(q), self._suppliers(q), self._departments(q)

        # dimension (what to group by)
        dim = None
        if re.search(r"\b(by|per|each|across|which|top|biggest|largest|highest|lowest|smallest|most|least)\b"
                     r"[^?]*\b(suppliers?|vendors?)\b", q) or re.search(r"\b(suppliers?|vendors?)\b.*\bby spend\b", q):
            dim = "supplier"
        elif re.search(r"\b(by|per|each|across|which|top|biggest|largest|highest|lowest|most|least)\b"
                       r"[^?]*\b(departments?|teams?|dept)\b", q):
            dim = "department"
        elif re.search(r"\b(by|per|each|across|which|top|biggest|largest|highest|lowest|most|least|breakdown)\b"
                       r"[^?]*\b(categor(y|ies))\b", q) or "category wise" in q or "category-wise" in q:
            dim = "category"
        elif re.search(r"\b(monthly|per month|by month|each month|month on month|month-wise|trend|over time)\b", q):
            dim = "month"
        elif re.search(r"\b(quarterly|by quarter|per quarter|each quarter)\b", q):
            dim = "quarter"
        # words like "marketing" / "hr" are both a department and a category -> disambiguate
        dept_words = {d.lower() for d in depts}
        if dept_words:
            def only_dept_word(cat):
                names = [cat.lower()] + CATEGORY_SYNONYMS.get(cat, [])
                return all(n in dept_words for n in names if _word_in(n, q))
            ambiguous = [c for c in cats if only_dept_word(c)]
            if ambiguous:
                if re.search(r"\b(department|dept|team)s?\b", q):
                    cats = [c for c in cats if c not in ambiguous]
                else:
                    depts = [d for d in depts if not any(_word_in(d.lower(), n) for c in ambiguous
                                                         for n in [c.lower()] + CATEGORY_SYNONYMS.get(c, []))]

        if dim == "department" and depts and not re.search(r"\b(by|per|each|across|which)\b[^?]*\b(departments|teams)\b", q):
            dim = None  # "spend by HR department" -> filter on HR, not a breakdown
        labels = []
        if cats and dim != "category":
            labels.append("category: " + ", ".join(cats))
            filters.append("category IN (" + ", ".join(f"'{_esc(c)}'" for c in cats) + ")")
        if sups and dim != "supplier":
            labels.append("supplier: " + ", ".join(sups))
            filters.append("supplier IN (" + ", ".join(f"'{_esc(s)}'" for s in sups) + ")")
        if depts and dim != "department":
            labels.append("department: " + ", ".join(depts))
            filters.append("department IN (" + ", ".join(f"'{_esc(d)}'" for d in depts) + ")")
        tf = self._time_filters(q)
        filters += tf
        if tf:
            labels.append("period: " + self._time_label(q))
        if re.search(r"\b(review|low confidence|uncertain|flagged)\b", q) and "needs_review" in self.db.df:
            filters.append("needs_review = 1")
            labels.append("flagged for review")
        where = (" WHERE " + " AND ".join(filters)) if filters else ""

        # metric
        if re.search(r"\bhow many (suppliers|vendors)\b|\bnumber of (suppliers|vendors)\b", q):
            return ParsedQuery(f"SELECT COUNT(DISTINCT supplier) AS suppliers FROM spend{where}",
                               "scalar", "distinct_suppliers", None, filters, labels)
        if re.search(r"\b(largest|biggest|highest|top) (invoices?|transactions?|purchases?|payments?)\b", q):
            n = int((re.search(r"\btop (\d+)\b", q) or [None, 10])[1])
            return ParsedQuery(
                f"SELECT invoice_date, supplier, description, category, amount_inr FROM spend{where} "
                f"ORDER BY amount_inr DESC LIMIT {n}", "rows", "rows", None, filters, labels)
        if re.search(r"\b(average|avg|mean)\b", q):
            metric, agg = "avg", "ROUND(AVG(amount_inr), 2) AS avg_amount_inr"
        elif re.search(r"\b(how many|number of|count)\b", q):
            metric, agg = "count", "COUNT(*) AS transactions"
        else:
            metric, agg = "sum", "ROUND(SUM(amount_inr), 2) AS total_spend_inr"

        if dim is None:
            extra = ", COUNT(*) AS transactions" if metric == "sum" else ""
            return ParsedQuery(f"SELECT {agg}{extra} FROM spend{where}", "scalar", metric, None, filters, labels)

        col = DIMENSIONS[dim][0]
        if dim in ("month", "quarter"):
            sql = f"SELECT {col}, {agg} FROM spend{where} GROUP BY {col} ORDER BY {col}"
            return ParsedQuery(sql, "trend", metric, dim, filters, labels)
        n = re.search(r"\b(?:top|bottom) (\d+)\b", q)
        limit = int(n.group(1)) if n else (5 if re.search(r"\btop\b", q) else 1 if re.search(
            r"\bwhich\b.*\b(most|highest|biggest|largest|least|lowest)\b", q) else 20)
        order = "ASC" if re.search(r"\b(lowest|least|smallest|bottom)\b", q) else "DESC"
        order_col = agg.split(" AS ")[1]
        if metric == "sum":
            agg += ", COUNT(*) AS transactions"
        sql = f"SELECT {col}, {agg} FROM spend{where} GROUP BY {col} ORDER BY {order_col} {order} LIMIT {limit}"
        return ParsedQuery(sql, "grouped", metric, dim, filters, labels)


def describe_result(pq: ParsedQuery, df) -> str:
    """Plain-English answer for a rule-based query result."""
    from src.agent.spend_db import format_crore_lakh, format_inr

    if df.empty:
        return "No matching spend records were found for that question."
    scope = ""
    if pq.labels:
        scope = " (" + "; ".join(pq.labels) + ")"
    if pq.kind == "scalar":
        row = df.iloc[0]
        if pq.metric == "sum":
            if not row["transactions"]:
                return f"No matching spend records were found{scope}."
            v = row["total_spend_inr"] or 0
            return f"Total spend is **{format_inr(v)}** ({format_crore_lakh(v)}) across {int(row['transactions'])} transactions{scope}."
        if pq.metric == "avg":
            return f"The average invoice line value is **{format_inr(row['avg_amount_inr'] or 0)}**{scope}."
        if pq.metric == "count":
            return f"There are **{int(row['transactions'])}** matching transactions{scope}."
        return f"There are **{int(row.iloc[0])}** distinct suppliers{scope}."
    if pq.kind == "rows":
        lines = [f"{i + 1}. {r.supplier} - {r.description} ({r.category}) - {format_inr(r.amount_inr)} on {r.invoice_date}"
                 for i, r in enumerate(df.itertuples())]
        return f"Largest transactions{scope}:\n" + "\n".join(lines)
    value_col = df.columns[1]
    fmt = (lambda v: f"{int(v)} txns") if pq.metric == "count" else format_inr
    if pq.kind == "trend":
        peak = df.loc[df[value_col].idxmax()]
        low = df.loc[df[value_col].idxmin()]
        return (f"{pq.dimension.title()}ly view over {len(df)} periods{scope}. Peak: **{peak.iloc[0]}** "
                f"({fmt(peak[value_col])}); lowest: {low.iloc[0]} ({fmt(low[value_col])}). See the chart/table below.")
    total = df[value_col].sum()
    lines = []
    for i, r in enumerate(df.itertuples(index=False)):
        share = f" ({r[1] / total:.0%} of shown)" if pq.metric == "sum" and len(df) > 1 and total else ""
        lines.append(f"{i + 1}. **{r[0]}** - {fmt(r[1])}{share}")
    head = {"sum": "Spend", "count": "Transaction count", "avg": "Average invoice value"}[pq.metric]
    return f"{head} by {pq.dimension}{scope}:\n" + "\n".join(lines)
