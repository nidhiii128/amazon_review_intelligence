"""Page 3: Product Insights -- the main page. Header + 4 tabs: Overview,
Aspect Analysis, Customer Reviews, Review Trust."""
import streamlit as st

from data_helpers import (
    display_name, get_product_overview, get_aspect_sentiment_for_asin,
    get_aspect_evidence, get_strengths_and_weaknesses, ai_summary_sentence,
    aspect_status, get_suspicious_summary_for_asin,
)
from theme import star_string, STATUS_COLOR_MAP, STATUS_EMOJI, PLACEHOLDER_IMAGE
from widgets import product_picker

ASPECT_ICONS = {
    "battery": "🔋", "camera": "📷", "display": "🖥️", "sound": "🔊",
    "performance": "⚡", "build_quality": "🛠️", "price_value": "💲",
    "connectivity": "📶", "size_comfort": "📏", "customer_service": "🎧",
}


def render(reviews, absa, products, suspicious):
    parent_asin = product_picker(reviews, products, "Choose a product", key="insights_product")

    name = display_name(products, parent_asin)
    row = products[products["parent_asin"] == parent_asin]
    brand = row.iloc[0]["brand"] if not row.empty and str(row.iloc[0].get("brand", "")).strip() else None
    image_url = row.iloc[0]["image_url"] if not row.empty else None

    header_col1, header_col2 = st.columns([1, 3])
    with header_col1:
        st.image(image_url or PLACEHOLDER_IMAGE, use_container_width=True)
    with header_col2:
        st.markdown(f"### {name}")
        overview = get_product_overview(reviews, parent_asin)
        st.markdown(f"{star_string(overview.get('avg_rating', 0))} **{overview.get('avg_rating', 0)}**"
                    f"&nbsp;&nbsp;📝 {overview.get('review_count', 0):,} Reviews")
        if brand:
            st.caption(f"Brand/Store: {brand}")

    aspect_table = get_aspect_sentiment_for_asin(absa, parent_asin)
    sw = get_strengths_and_weaknesses(aspect_table)

    tab_overview, tab_aspects, tab_reviews, tab_trust = st.tabs(
        ["Overview", "Aspect Analysis", "Customer Reviews", "🛡️ Review Trust"]
    )

    with tab_overview:
        st.markdown("#### 🧠 AI Summary")
        st.info(ai_summary_sentence(overview, sw["strengths"], sw["weaknesses"]))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 👍 What Customers Like")
            if sw["strengths"]:
                for s in sw["strengths"]:
                    st.success(s.replace("_", " ").title())
            else:
                st.caption("Not enough data.")
        with col_b:
            st.markdown("#### ⚠️ Common Concerns")
            if sw["weaknesses"]:
                for w in sw["weaknesses"]:
                    st.error(w.replace("_", " ").title())
            else:
                st.caption("Not enough data.")

    with tab_aspects:
        st.markdown("#### Detected aspects for this product")
        if aspect_table.empty:
            st.info("Not enough aspect mentions for this product in the sample.")
        else:
            for aspect, r in aspect_table.iterrows():
                status = aspect_status(r)
                icon = ASPECT_ICONS.get(aspect, "🔹")
                label = aspect.replace("_", " ").title()
                with st.expander(f"{icon} {label} — {STATUS_EMOJI[status]} {status.title()} "
                                  f"({int(r['mentions'])} mentions)"):
                    st.caption(f"{r['positive']}% positive · {r['neutral']}% neutral · {r['negative']}% negative")
                    evidence = get_aspect_evidence(absa, parent_asin, aspect)
                    if evidence.empty:
                        st.caption("No supporting sentences captured for this aspect.")
                    for _, ev in evidence.iterrows():
                        st.markdown(
                            f'<div class="evidence-card">{STATUS_EMOJI.get(ev["aspect_sentiment"], "⚪")} '
                            f'{ev["evidence"]}</div>',
                            unsafe_allow_html=True,
                        )

    with tab_reviews:
        st.markdown("#### Customer Reviews")
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
        st.dataframe(
            filtered[["rating", "sentiment_label", "verified_purchase", "clean_text"]].head(200),
            use_container_width=True,
        )

    with tab_trust:
        st.markdown("#### Review Trust Analysis")
        if suspicious.empty:
            st.info("Suspicious-review detection hasn't been generated yet -- run "
                     "src/advanced_ml/suspicious_reviews.py.")
        else:
            summary = get_suspicious_summary_for_asin(suspicious, reviews, parent_asin)
            st.caption(f"Reviews Analysed: {summary['total']:,}")
            c1, c2 = st.columns(2)
            c1.metric("🟢 Normal Pattern", summary["normal"])
            c2.metric("🟡 Potentially Suspicious", summary["suspicious"])

            if summary["suspicious"] == 0:
                st.success("No potentially suspicious reviews detected for this product.")
            else:
                st.markdown("##### ⚠️ Potentially Suspicious")
                for _, r in summary["flagged"].head(15).iterrows():
                    reasons = r["reasons"] if isinstance(r["reasons"], str) else ""
                    st.markdown(
                        f'<div class="suspicious-card"><b>Score: {r["suspicion_score"]:.2f}</b><br>'
                        f'Possible signals: {reasons}</div>',
                        unsafe_allow_html=True,
                    )
            st.caption(
                "⚠️ Potentially suspicious based on automated pattern analysis -- "
                "this is not a claim that any specific review is fake."
            )
