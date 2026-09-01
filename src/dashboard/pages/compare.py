"""Page 4: Compare Products -- computed comparison table + a template-
generated summary sentence (never an LLM call, so it can't invent facts)."""
import streamlit as st

from data_helpers import get_comparison_table, ai_comparison_sentence
from widgets import product_picker


def render(reviews, absa, products):
    col1, col2 = st.columns(2)
    with col1:
        parent_asin_a = product_picker(reviews, products, "Product A", key="compare_a")
    with col2:
        parent_asin_b = product_picker(reviews, products, "Product B", key="compare_b")

    st.markdown("---")
    comparison = get_comparison_table(reviews, absa, products, parent_asin_a, parent_asin_b)

    display_table = comparison.rename(columns={
        "name": "Product", "rating": "Rating", "reviews": "Reviews",
        "positive_pct": "% Positive", "negative_pct": "% Negative", "main_concern": "Main Concern",
    }).T.astype(str)
    st.dataframe(display_table, use_container_width=True)

    st.markdown("#### 🤖 AI Comparison")
    st.info(ai_comparison_sentence(comparison))
