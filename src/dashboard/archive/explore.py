"""Page 2: Explore Products -- search/browse, identified by product name."""
import streamlit as st

from data_helpers import get_top_products, search_products_by_name
from widgets import render_product_card, go_to


def render(reviews, products):
    st.subheader("🔎 Search a Product")

    query = st.text_input("Search by product name...", placeholder="e.g. wireless earbuds")
    top_products = get_top_products(reviews, products, n=500)
    if query:
        matches = search_products_by_name(top_products, query)
    else:
        matches = top_products

    if matches.empty:
        st.info("No products match that search.")
        return

    st.caption(f"{len(matches):,} products found")
    cols = st.columns(4)
    for i, row in enumerate(matches.head(40).itertuples()):
        with cols[i % 4]:
            render_product_card(row)
            st.button("View AI Insights", key=f"explore_{row.parent_asin}", use_container_width=True,
                      on_click=go_to, args=("📊 Product Insights",),
                      kwargs={"selected_insights_product": row.parent_asin})
