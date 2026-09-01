"""Page 1: Home -- static landing page, no auth."""
import streamlit as st

PIPELINE_STEPS = [
    "Reviews", "NLP", "Deep Learning", "Advanced ML",
    "Product Intelligence", "Semantic Search", "RAG",
]


def render(reviews):
    st.markdown(
        """
        <div style="text-align:center; padding: 24px 0 8px 0;">
            <div style="font-size:40px; font-weight:800;">🧠 REVIEWIQ</div>
            <div style="font-size:18px; color:#4b5563; margin-top:4px;">
                AI-Based Product Review Intelligence &amp; Consumer Insight System
            </div>
            <div style="font-size:15px; color:#6b7280; margin-top:10px;">
                Turn thousands of customer reviews into actionable insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        b1, b2 = st.columns(2)
        if b1.button("🛒 Explore as Buyer", use_container_width=True):
            st.session_state["page"] = "🔎 Explore Products"
            st.rerun()
        if b2.button("📊 Business Insights", use_container_width=True):
            st.session_state["page"] = "📈 Business Intelligence"
            st.rerun()

    st.markdown("---")
    st.markdown("#### Project pipeline")
    cols = st.columns(len(PIPELINE_STEPS))
    for c, step in zip(cols, PIPELINE_STEPS):
        c.markdown(
            f"<div style='text-align:center; background:white; border:1px solid #e4e7eb; "
            f"border-radius:8px; padding:10px 4px; font-size:12px; font-weight:600;'>{step}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Reviews analyzed", f"{len(reviews):,}")
    m2.metric("Products", f"{reviews['parent_asin'].nunique():,}")
    m3.metric("Category", "Electronics")
