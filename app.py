"""Streamlit web app: categorize procurement data + chat with the spend agent.

Run:  streamlit run app.py
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.agent.agent import SpendAgent
from src.agent.llm_backends import LLMError, get_backend
from src.agent.spend_db import SpendDatabase, format_crore_lakh, format_inr
from src.categorize import categorize_df

st.set_page_config(page_title="Spend Categorization Agent", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier(kind: str):
    if kind == "llm":
        from src.llm_classifier import LLMSpendClassifier
        return LLMSpendClassifier()
    from src.baseline import BaselineClassifier
    return BaselineClassifier()


@st.cache_data(show_spinner=False)
def default_data() -> pd.DataFrame | None:
    if config.CATEGORIZED_PATH.exists():
        return pd.read_csv(config.CATEGORIZED_PATH)
    return None


def sorted_bar(series: pd.Series, horizontal: bool = False):
    """Bar chart sorted by value (st.bar_chart sorts labels alphabetically)."""
    d = series.reset_index()
    d.columns = ["label", "value"]
    if horizontal:
        enc = dict(y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                   x=alt.X("value:Q", title="Rs"))
    else:
        enc = dict(x=alt.X("label:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=160)),
                   y=alt.Y("value:Q", title=None))
    chart = alt.Chart(d).mark_bar().encode(**enc, tooltip=["label", alt.Tooltip("value:Q", format=",.0f")])
    st.altair_chart(chart.properties(height=max(260, 26 * len(d)) if horizontal else 320), width="stretch")


def llm_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and (config.ADAPTER_DIR / "adapter_config.json").exists()
    except ImportError:
        return False


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Settings")
if "spend_df" not in st.session_state:
    st.session_state.spend_df = default_data()

backend_name = st.sidebar.selectbox(
    "Agent language model", ["rules", "gemini", "groq", "ollama"],
    index=["rules", "gemini", "groq", "ollama"].index(config.AGENT_BACKEND)
    if config.AGENT_BACKEND in ("rules", "gemini", "groq", "ollama") else 0,
    help="rules = offline parser, no key needed. gemini/groq need a free API key. ollama runs Llama 3 locally.")
api_key = ""
if backend_name in ("gemini", "groq"):
    env_key = config.GEMINI_API_KEY if backend_name == "gemini" else config.GROQ_API_KEY
    api_key = st.sidebar.text_input(f"{backend_name.title()} API key", value=env_key, type="password",
                                    help="Stored only in this browser session. You can also put it in the .env file.")
st.sidebar.caption("Classifier for new uploads")
clf_options = ["Baseline (TF-IDF, CPU)"] + (["Fine-tuned Llama-3 8B (GPU)"] if llm_available() else [])
clf_choice = st.sidebar.radio("Classifier", clf_options, label_visibility="collapsed")
if not llm_available():
    st.sidebar.info("Running the fine-tuned Llama-3 here needs an NVIDIA GPU and the adapter in "
                    "models/llama3-spend-qlora. Without it, uploads use the baseline. To get Llama-3 predictions, "
                    "run the Colab notebook and unzip results.zip into the project folder.")

st.title("📊 Automated Spend Categorization & AI Agent")
st.caption("Llama-3 8B fine-tuned with QLoRA classifies messy procurement lines into standard spend categories; "
           "a natural-language agent answers spend questions.")

tab_cat, tab_dash, tab_chat, tab_model = st.tabs(["1 · Categorize", "2 · Dashboard", "3 · Ask the agent",
                                                   "4 · Model results"])

# ------------------------------------------------------------------ tab 1: categorize
with tab_cat:
    st.subheader("Categorize procurement data")
    st.write("Upload a CSV with at least `supplier_name` and `line_description` columns "
             "(optional: `department`, `amount_inr`, `invoice_date`).")
    up = st.file_uploader("Procurement CSV", type=["csv"])
    c1, c2 = st.columns(2)
    run_upload = c1.button("Categorize uploaded file", disabled=up is None, type="primary")
    run_sample = c2.button("Categorize the sample dataset (5,000 rows)")
    if run_upload or run_sample:
        raw = pd.read_csv(up) if run_upload else pd.read_csv(config.RAW_DATA_PATH)
        kind = "llm" if clf_choice.startswith("Fine-tuned") else "baseline"
        try:
            with st.spinner(f"Classifying {len(raw)} rows..."):
                out = categorize_df(raw, kind, classifier=load_classifier(kind))
            st.session_state.spend_df = out
            st.session_state.pop("agent_key", None)
            st.success(f"Done: {len(out)} rows categorized, {int(out['needs_review'].sum())} flagged for review.")
        except ValueError as e:
            st.error(str(e))
    df = st.session_state.spend_df
    if df is not None:
        show = [c for c in ["invoice_date", "supplier_name", "line_description", "department", "amount_inr",
                            "predicted_category", "confidence", "needs_review", "true_category"] if c in df.columns]
        only_review = st.checkbox("Show only rows that need human review")
        view = df[df["needs_review"]] if only_review and "needs_review" in df else df
        st.dataframe(view[show], width="stretch", height=420)
        st.download_button("Download categorized CSV", df.to_csv(index=False).encode(), "categorized_spend.csv",
                           "text/csv")
    else:
        st.info("No categorized data yet - categorize the sample dataset or upload a file.")

# ------------------------------------------------------------------ tab 2: dashboard
with tab_dash:
    df = st.session_state.spend_df
    if df is None:
        st.info("Categorize some data first.")
    else:
        cat_col = "predicted_category" if "predicted_category" in df else "category"
        d = df.copy()
        d["month"] = pd.to_datetime(d.get("invoice_date"), errors="coerce").dt.strftime("%Y-%m")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total spend", format_crore_lakh(d["amount_inr"].sum()))
        k2.metric("Invoice lines", f"{len(d):,}")
        k3.metric("Suppliers", f"{d['supplier_name'].nunique():,}")
        if "needs_review" in d:
            k4.metric("Need review", f"{int(d['needs_review'].sum()):,}")
        if "classifier" in d:
            st.caption(f"Categories predicted by: **{', '.join(d['classifier'].dropna().unique())}**")
        st.markdown("**Spend by category (Rs)**")
        sorted_bar(d.groupby(cat_col)["amount_inr"].sum(), horizontal=True)
        a, b = st.columns(2)
        a.markdown("**Monthly spend (Rs)**")
        a.line_chart(d.groupby("month")["amount_inr"].sum())
        b.markdown("**Top 10 suppliers**")
        top = d.groupby("supplier_name")["amount_inr"].sum().sort_values(ascending=False).head(10)
        b.dataframe(top.map(format_inr).rename("spend"), width="stretch")

# ------------------------------------------------------------------ tab 3: chat
with tab_chat:
    df = st.session_state.spend_df
    if df is None:
        st.info("Categorize some data first.")
    else:
        key = (backend_name, api_key, id(df))
        if st.session_state.get("agent_key") != key:
            try:
                backend = get_backend(backend_name, api_key or None)
            except LLMError as e:
                st.warning(f"{e} Using the offline rule-based agent instead.")
                backend = None
            st.session_state.agent = SpendAgent(SpendDatabase(df), backend)
            st.session_state.agent_key = key
            st.session_state.messages = []
        agent: SpendAgent = st.session_state.agent
        st.caption(f"Agent backend: **{agent.backend.name if agent.backend else 'rules (offline)'}**. "
                   "Try: *top 5 suppliers by spend*, *monthly trend of travel spend*, "
                   "*how much did we spend on software in Q4 2025?*")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("table") is not None:
                    st.dataframe(m["table"], width="stretch")
        q = st.chat_input("Ask about your spend...")
        if q:
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    r = agent.ask(q)
                st.markdown(r.answer)
                tbl = r.table if r.table is not None and len(r.table) > 1 else None
                if tbl is not None:
                    num = tbl.select_dtypes("number").columns
                    if len(tbl.columns) >= 2 and len(num) >= 1 and tbl.columns[0] not in num:
                        chart = tbl.set_index(tbl.columns[0])[num[0]]
                        if tbl.columns[0] in ("month", "quarter"):
                            st.line_chart(chart)
                        else:
                            sorted_bar(chart)
                    st.dataframe(tbl, width="stretch")
                if r.sql:
                    with st.expander("SQL the agent ran"):
                        st.code(r.sql, language="sql")
                for n in r.notes:
                    st.caption(n)
            st.session_state.messages.append({"role": "assistant", "content": r.answer, "table": tbl})

# ------------------------------------------------------------------ tab 4: model results
with tab_model:
    st.subheader("Classifier evaluation on the held-out test set")
    rows = []
    for name in ["baseline", "llm_zero_shot", "llm_qlora"]:
        p = config.REPORTS_DIR / f"metrics_{name}.json"
        if p.exists():
            m = json.loads(p.read_text())
            rows.append({"model": m.get("model", name), "accuracy": m["accuracy"], "macro F1": m["macro_f1"],
                         "seen items": m.get("accuracy_seen_items"), "unseen items": m.get("accuracy_unseen_items"),
                         "test rows": m["n"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("'Unseen items' = products that never appeared in training data (tests generalisation).")
    else:
        st.info("No reports yet. Run `python -m src.baseline` and the Colab notebook.")
    imgs = [p for p in ["training_loss.png", "confusion_llm_qlora.png", "confusion_baseline.png"]
            if (config.REPORTS_DIR / p).exists()]
    for p in imgs:
        st.image(str(config.REPORTS_DIR / p), caption=p, width=700)
