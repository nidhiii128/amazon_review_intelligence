"""Seller Insights: what to fix first (complaint ranking) and how much to
trust the reviews (helpfulness and suspicious-review views)."""
import html
import pathlib

import joblib
import streamlit as st

from data_helpers import (
    get_product_overview, get_corpus_top_complaints,
    load_suspicious, build_suspicious_table,
)
from widgets import product_select

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def _load_helpfulness_model(path_str: str):
    return joblib.load(path_str)


@st.cache_resource(show_spinner="Scoring review helpfulness (first time only) ...")
def _helpfulness_scores(_reviews):
    """Predicted helpfulness per review. Building the features scores VADER
    on every review (about 25 s), so the result is saved next to the other
    processed data and reused by later sessions."""
    import pandas as pd
    from data_helpers import DATA_DIR
    path = DATA_DIR / "helpfulness_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    from helpfulness_xgboost import build_features, FEATURE_COLS
    model = _load_helpfulness_model(str(MODELS_DIR / "xgboost_helpfulness.joblib"))
    featured = build_features(_reviews)
    scores = featured[["review_id"]].assign(predicted_helpful_proba=model.predict_proba(featured[FEATURE_COLS])[:, 1])
    scores.to_csv(path, index=False)
    return scores


def _complaint_bars(complaints) -> str:
    top = max(float(complaints["negative_pct"].max()), 1.0)
    rows = []
    for rank, (aspect, r) in enumerate(complaints.iterrows(), 1):
        label = html.escape(aspect.replace("_", " ").title())
        width = r["negative_pct"] / top * 100
        rows.append(
            f'<div style="display:grid;grid-template-columns:26px 150px 1fr 250px;gap:12px;align-items:center;'
            f'background:#fff;border-radius:12px;padding:12px 16px;margin-bottom:8px">'
            f'<b style="color:#8a92a3">{rank}</b><b>{label}</b>'
            f'<div style="background:#f1f2f5;border-radius:6px;height:10px"><div style="width:{width:.0f}%;height:10px;'
            f'border-radius:6px;background:linear-gradient(90deg,#f5a300,#d03b3b)"></div></div>'
            f'<span style="color:#6b7280;font-size:13px"><b style="color:#d03b3b">{r["negative_pct"]}%</b> negative · '
            f'{int(r["mentions"]):,} mentions</span></div>'
        )
    return "".join(rows)


def render(reviews, absa, table):
    scope = st.radio("Analysis scope", ["All products", "One product"], horizontal=True)
    scoped = None
    if scope == "One product":
        scoped = product_select(table, "Product", key="seller", shared=False)

    tab_overview, tab_complaints, tab_quality = st.tabs(["Overview", "🔥 What to fix first", "🛡️ Review quality"])

    with tab_overview:
        if scoped is None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total reviews", f"{len(reviews):,}")
            c2.metric("Products", f"{reviews['parent_asin'].nunique():,}")
            c3.metric("Average rating", f"{reviews['rating'].mean():.2f}")
            c4.metric("% Positive", f"{(reviews['sentiment_label'] == 'positive').mean() * 100:.1f}%")
            st.markdown("##### Sentiment distribution")
            st.bar_chart(reviews["sentiment_label"].value_counts(), color="#f5a300")
        else:
            overview = get_product_overview(reviews, scoped)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rating", f"{overview.get('avg_rating', 0)} / 5")
            c2.metric("Reviews", overview.get("review_count", 0))
            c3.metric("% Positive", f"{overview.get('positive_pct', 0)}%")

    with tab_complaints:
        st.caption("Aspects ranked by the share of negative mentions in customer reviews.")
        complaints = get_corpus_top_complaints(absa, top_n=10, parent_asin=scoped)
        if complaints.empty:
            st.info("Not enough aspect mentions for this selection.")
        else:
            st.markdown(_complaint_bars(complaints), unsafe_allow_html=True)

    with tab_quality:
        st.markdown("##### 👍 Most informative reviews")
        st.caption("Reviews the helpfulness model (XGBoost) expects shoppers to find most useful.")
        model_path = MODELS_DIR / "xgboost_helpfulness.joblib"
        if model_path.exists():
            scores = _helpfulness_scores(reviews)
            top_helpful = (scores.sort_values("predicted_helpful_proba", ascending=False).head(10)
                           .merge(reviews[["review_id", "rating", "clean_text"]], on="review_id"))
            st.dataframe(top_helpful[["rating", "predicted_helpful_proba", "clean_text"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("XGBoost model not found. Run src/advanced_ml/helpfulness_xgboost.py first.")

        st.markdown("##### 🛡️ Potentially suspicious reviews")
        suspicious = load_suspicious()
        if suspicious.empty:
            st.info("Not generated yet. Run src/advanced_ml/suspicious_reviews.py.")
        else:
            threshold = st.slider(
                "Suspicion score threshold (reviews at or above this score are red-flagged 🚩)",
                min_value=0.0, max_value=1.0, value=0.6, step=0.05, key="bi_trust_threshold",
            )
            flags = build_suspicious_table(suspicious, reviews, threshold, parent_asin=scoped)
            c1, c2 = st.columns(2)
            c1.metric("Reviews analysed", f"{len(flags):,}")
            c2.metric("🚩 Potentially suspicious", f"{int(flags['red_flag'].sum()):,}")
            shown = flags[flags["red_flag"]].rename(columns={
                "red_flag": "🚩 Flag", "suspicion_score": "Score", "rating": "Rating",
                "clean_text": "Review", "reasons": "Possible Signals",
            })[["🚩 Flag", "Score", "Rating", "Review", "Possible Signals"]]
            st.dataframe(
                shown.head(50), use_container_width=True, hide_index=True,
                column_config={
                    "🚩 Flag": st.column_config.CheckboxColumn(disabled=True),
                    "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                    "Review": st.column_config.TextColumn(width="large"),
                },
            )
            st.caption("Unsupervised anomaly detection (Isolation Forest), not a claim that any review is fake.")
