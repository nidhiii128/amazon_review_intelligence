"""Compare two products. The summary sentence is built from computed
numbers by a template, never by a language model."""
import streamlit as st

from data_helpers import get_comparison_table, ai_comparison_sentence
from widgets import product_select, product_card_html


def render(reviews, absa, products, table):
    col_a, col_b = st.columns(2)
    with col_a:
        asin_a = product_select(table, "Product A", key="compare_a")
    ranked = table.sort_values("review_count", ascending=False)["parent_asin"].tolist()
    default_b = next((a for a in ranked if a != asin_a), ranked[0])
    with col_b:
        asin_b = product_select(table, "Product B", key="compare_b", default_asin=default_b, shared=False)

    for col, asin in zip(st.columns(2), (asin_a, asin_b)):
        with col:
            st.markdown(product_card_html(table[table["parent_asin"] == asin].iloc[0]), unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Side by side</div>', unsafe_allow_html=True)
    comparison = get_comparison_table(reviews, absa, products, asin_a, asin_b)
    display_table = comparison.rename(columns={
        "name": "Product", "rating": "Rating", "reviews": "Reviews",
        "positive_pct": "% Positive", "negative_pct": "% Negative", "main_concern": "Main Concern",
    }).T.astype(str)
    st.dataframe(display_table, use_container_width=True)
    st.info(ai_comparison_sentence(comparison))
