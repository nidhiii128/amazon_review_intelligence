"""Shared interactive widgets (product picker / card grid) reused across
Explore, Insights, and Compare pages."""
import streamlit as st

from theme import star_string, PLACEHOLDER_IMAGE


def render_product_grid(top_products, session_key: str, n: int = 8):
    """A clickable grid of product cards (name + image + rating + review
    count) -- selecting a card updates session_state[session_key] and
    reruns the app."""
    rows = top_products.head(n)
    cols = st.columns(4)
    for i, row in enumerate(rows.itertuples()):
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
            if st.button("View AI Insights", key=f"{session_key}_{row.parent_asin}"):
                st.session_state[session_key] = row.parent_asin


def product_picker(reviews, products, label: str, key: str):
    """Card grid for browsing top products, plus a name search box and a
    full dropdown as a fallback for finding any product. Returns the
    selected parent_asin."""
    from data_helpers import get_top_products, search_products_by_name

    top_products = get_top_products(reviews, products, n=200)
    session_key = f"selected_{key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = top_products.iloc[0]["parent_asin"]

    st.markdown(f"**{label}**")
    search = st.text_input("🔎 Search by product name", key=f"search_{key}", placeholder="Type part of a product name...")
    filtered = search_products_by_name(top_products, search)
    render_product_grid(filtered, session_key)

    options = top_products["parent_asin"].tolist()
    current = st.session_state[session_key]
    idx = options.index(current) if current in options else 0
    lookup = top_products.set_index("parent_asin")
    chosen = st.selectbox(
        "Or pick from the full list", options, index=idx, key=f"select_{key}",
        format_func=lambda pa: f"{lookup.loc[pa, 'product_name'][:60]}  "
                                f"({lookup.loc[pa, 'avg_rating']}★, {lookup.loc[pa, 'review_count']} reviews)",
    )
    st.session_state[session_key] = chosen
    return st.session_state[session_key]
