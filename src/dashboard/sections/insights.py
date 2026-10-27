"""Product Insights: one product, four tabs (Overview, Aspect Analysis,
Customer Reviews, Review Trust)."""
import html
import re

import streamlit as st

from data_helpers import (
    get_product_overview, get_aspect_sentiment_for_asin, get_aspect_evidence,
    get_strengths_and_weaknesses, ai_summary_sentence, aspect_status, build_suspicious_table,
)
from theme import star_string, STATUS_EMOJI, image_style
from widgets import product_select, go_to, PAGE_ASK, PAGE_COMPARE

ASPECT_ICONS = {
    "battery": "🔋", "camera": "📷", "display": "🖥️", "sound": "🔊",
    "performance": "⚡", "build_quality": "🛠️", "price_value": "💲",
    "connectivity": "📶", "size_comfort": "📏", "customer_service": "🎧",
}


def render(reviews, absa, table, suspicious, lstm_preds):
    parent_asin = product_select(table, "Product", key="insights")
    row = table[table["parent_asin"] == parent_asin].iloc[0]
    overview = get_product_overview(reviews, parent_asin)
    brand = str(row["brand"]).strip() if isinstance(row["brand"], str) else ""
    neutral_pct = round(100 - overview.get("positive_pct", 0) - overview.get("negative_pct", 0), 1)

    detail = (
        f"""<div class="detail"><div class="dimg" style="{image_style(row['image_url'])}"></div>
        <div><h3>{html.escape(str(row['product_name']))}</h3>
        <div class="stars">{star_string(overview.get('avg_rating', 0))} {overview.get('avg_rating', 0)}</div>
        <div class="chips" style="margin-top:12px">
          <span class="chip"><b>{overview.get('review_count', 0):,}</b> reviews</span>
          <span class="chip"><b>{overview.get('positive_pct', 0)}%</b> positive</span>
          <span class="chip"><b>{overview.get('verified_pct', 0)}%</b> verified purchases</span>
          {f'<span class="chip">{html.escape(brand)}</span>' if brand else ''}
        </div></div></div>"""
    )
    st.markdown(re.sub(r"\n\s*", "", detail), unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.7, 1.2, 4])
    b1.button("💬 Ask about this product", use_container_width=True, on_click=go_to, args=(PAGE_ASK,))
    b2.button("⚖️ Compare", use_container_width=True, on_click=go_to, args=(PAGE_COMPARE,))

    aspect_table = get_aspect_sentiment_for_asin(absa, parent_asin)
    sw = get_strengths_and_weaknesses(aspect_table)

    tab_overview, tab_aspects, tab_reviews, tab_trust = st.tabs(
        ["Overview", "Aspect Analysis", "Customer Reviews", "🛡️ Review Trust"]
    )

    with tab_overview:
        st.info(ai_summary_sentence(overview, sw["strengths"], sw["weaknesses"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Positive", f"{overview.get('positive_pct', 0)}%")
        m2.metric("🟡 Neutral", f"{neutral_pct}%")
        m3.metric("🔴 Negative", f"{overview.get('negative_pct', 0)}%")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 👍 What customers like")
            for s in sw["strengths"]:
                st.success(s.replace("_", " ").title())
            if not sw["strengths"]:
                st.caption("Not enough data.")
        with col_b:
            st.markdown("#### ⚠️ Common concerns")
            for w in sw["weaknesses"]:
                st.error(w.replace("_", " ").title())
            if not sw["weaknesses"]:
                st.caption("Not enough data.")

    with tab_aspects:
        if aspect_table.empty:
            st.info("Not enough aspect mentions for this product in the sample.")
        else:
            st.caption("Open an aspect to see the exact review sentences behind its score.")
            for aspect, r in aspect_table.iterrows():
                status = aspect_status(r)
                label = aspect.replace("_", " ").title()
                with st.expander(f"{ASPECT_ICONS.get(aspect, '🔹')} {label}: {STATUS_EMOJI[status]} {status.title()} "
                                 f"({int(r['mentions'])} mentions)"):
                    st.caption(f"{r['positive']}% positive · {r['neutral']}% neutral · {r['negative']}% negative")
                    evidence = get_aspect_evidence(absa, parent_asin, aspect)
                    if evidence.empty:
                        st.caption("No supporting sentences captured for this aspect.")
                    for _, ev in evidence.iterrows():
                        st.markdown(
                            f'<div class="evidence-card">{STATUS_EMOJI.get(ev["aspect_sentiment"], "⚪")} '
                            f'{html.escape(str(ev["evidence"]))}</div>',
                            unsafe_allow_html=True,
                        )

    with tab_reviews:
        sub = reviews[reviews["parent_asin"] == parent_asin]
        susp_ids = set()
        if not suspicious.empty:
            susp_ids = set(suspicious[suspicious["suspicion_label"] == "potentially_suspicious"]["review_id"])

        f1, f2, f3, f4 = st.columns(4)
        sentiment_filter = f1.selectbox("Sentiment", ["All", "positive", "neutral", "negative"])
        rating_filter = f2.selectbox("Rating", ["All", "5", "4", "3", "2", "1"])
        verified_only = f3.checkbox("Verified purchase only")
        hide_suspicious = f4.checkbox("Hide suspicious")

        filtered = sub
        if sentiment_filter != "All":
            filtered = filtered[filtered["sentiment_label"] == sentiment_filter]
        if rating_filter != "All":
            filtered = filtered[filtered["rating"] == float(rating_filter)]
        if verified_only:
            filtered = filtered[filtered["verified_purchase"]]
        if hide_suspicious and susp_ids:
            filtered = filtered[~filtered["review_id"].isin(susp_ids)]

        st.caption(f"{len(filtered):,} of {len(sub):,} reviews shown")
        shown = filtered
        columns = ["rating", "sentiment_label", "verified_purchase", "clean_text"]
        if not lstm_preds.empty:
            shown = filtered.merge(lstm_preds, on="review_id", how="left")
            columns = ["rating", "sentiment_label", "lstm_label", "lstm_confidence", "verified_purchase", "clean_text"]
        st.dataframe(
            shown[columns].rename(columns={
                "sentiment_label": "rating-based label", "lstm_label": "🧠 LSTM prediction",
                "lstm_confidence": "LSTM confidence",
            }).head(200),
            use_container_width=True, hide_index=True,
        )

    with tab_trust:
        if suspicious.empty:
            st.info("Suspicious-review detection hasn't been generated yet. Run "
                    "src/advanced_ml/suspicious_reviews.py.")
        else:
            threshold = st.slider(
                "Suspicion score threshold (reviews at or above this score are red-flagged 🚩)",
                min_value=0.0, max_value=1.0, value=0.6, step=0.05, key="trust_threshold",
            )
            flags = build_suspicious_table(suspicious, reviews, threshold, parent_asin=parent_asin)

            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews analysed", f"{len(flags):,}")
            c2.metric("🟢 Normal pattern", int((~flags["red_flag"]).sum()))
            c3.metric("🚩 Potentially suspicious", int(flags["red_flag"].sum()))

            display = flags.rename(columns={
                "red_flag": "🚩 Flag", "suspicion_score": "Score", "rating": "Rating",
                "clean_text": "Review", "reasons": "Possible Signals",
            })[["🚩 Flag", "Score", "Rating", "Review", "Possible Signals"]]
            st.dataframe(
                display, use_container_width=True, hide_index=True,
                column_config={
                    "🚩 Flag": st.column_config.CheckboxColumn(disabled=True),
                    "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                    "Review": st.column_config.TextColumn(width="large"),
                },
            )
            st.caption("⚠️ Potentially suspicious based on automated pattern analysis. This is not a claim "
                       "that any specific review is fake.")
