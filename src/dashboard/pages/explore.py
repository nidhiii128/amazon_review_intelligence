"""Page 2: Explore Products -- search/browse, identified by product name."""
import streamlit as st

from data_helpers import get_top_products, search_products_by_name
from theme import star_string, PLACEHOLDER_IMAGE


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
            image_url = getattr(row, "image_url", None) or PLACEHOLDER_IMAGE
            st.image(image_url, use_container_width=True)
            st.markdown(
                f"""<div class="product-card">
                    <div class="pname">{row.product_name[:60]}</div>
                    <div class="pstars">{star_string(row.avg_rating)} {row.avg_rating}</div>
                    <div class="pmeta">{row.review_count:,} reviews</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("View AI Insights", key=f"explore_{row.parent_asin}"):
                st.session_state["selected_insights_product"] = row.parent_asin
                st.session_state["page"] = "📊 Product Insights"
                st.rerun()
