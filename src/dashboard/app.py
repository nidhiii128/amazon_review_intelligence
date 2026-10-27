"""
ReviewIQ dashboard: a shop-style app built for the live demo.

  Shop (search + shelves) -> Product Insights -> Compare -> Ask ReviewIQ,
  plus Seller Insights for the business view.

Run with:  streamlit run src/dashboard/app.py
"""
import sys
import pathlib

import streamlit as st

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "dashboard"))
sys.path.insert(0, str(BASE_DIR / "src" / "rag_chatbot"))
sys.path.insert(0, str(BASE_DIR / "src" / "advanced_ml"))

from data_helpers import load_reviews, load_absa, load_products, load_suspicious, load_lstm_predictions, get_product_table
from theme import inject_css, render_sidebar_brand, image_style
from widgets import PAGES, PAGE_HOME, PAGE_INSIGHTS, PAGE_COMPARE, PAGE_ASK, PAGE_SELLER, go_to, open_product
from sections import home, insights, compare, ask_reviewiq, business_intelligence

st.set_page_config(page_title="ReviewIQ", page_icon="🛒", layout="wide")


# cache_resource hands every rerun the same in-memory DataFrame; cache_data would
# hash the 50,000-row table and copy it on every click (seconds per interaction).
# The underscore-prefixed arguments are deliberately excluded from hashing.
@st.cache_resource
def _load_reviews():
    return load_reviews()


@st.cache_resource
def _load_absa(_reviews):
    return load_absa(_reviews)


@st.cache_resource
def _load_products():
    return load_products()


@st.cache_resource
def _load_suspicious():
    return load_suspicious()


@st.cache_resource
def _load_lstm_predictions():
    return load_lstm_predictions()


@st.cache_resource
def _product_table(_reviews, _products):
    return get_product_table(_reviews, _products)


def _search_changed():
    st.session_state["nav"] = PAGE_HOME


def render_sidebar(reviews, table):
    render_sidebar_brand()
    st.session_state.setdefault("nav", PAGE_HOME)
    page = st.sidebar.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")

    st.sidebar.markdown('<div class="sb-label">Search your products</div>', unsafe_allow_html=True)
    st.sidebar.text_input("Search", key="q", placeholder="e.g. echo, earbuds, charger",
                          label_visibility="collapsed", on_change=_search_changed)

    st.sidebar.markdown('<div class="sb-label">Recommended products</div>', unsafe_allow_html=True)
    rec = table[table["review_count"] >= 15].sort_values(
        ["avg_rating", "positive_pct", "review_count"], ascending=False).head(4)
    for col, row in zip(st.sidebar.columns(4), rec.itertuples()):
        with col:
            with st.container(key=f"thumb_{row.parent_asin}"):
                st.markdown(f'<div class="thumb" style="{image_style(row.image_url)}"></div>'
                            f'<div class="thumb-cap">{row.avg_rating}★</div>', unsafe_allow_html=True)
                st.button("open", key=f"thumbbtn_{row.parent_asin}", on_click=open_product, args=(row.parent_asin,))

    st.sidebar.markdown(
        f'<div class="sb-stats"><div class="big">{len(reviews):,}</div>'
        f'<div class="sub">reviews analysed across {len(table):,} products</div></div>',
        unsafe_allow_html=True,
    )
    return page


def main():
    inject_css()
    reviews = _load_reviews()
    absa = _load_absa(reviews)
    products = _load_products()
    suspicious = _load_suspicious()
    lstm_preds = _load_lstm_predictions()
    table = _product_table(reviews, products)

    page = render_sidebar(reviews, table)

    if page == PAGE_HOME:
        home.render(reviews, absa, table)
    elif page == PAGE_INSIGHTS:
        insights.render(reviews, absa, table, suspicious, lstm_preds)
    elif page == PAGE_COMPARE:
        compare.render(reviews, absa, products, table)
    elif page == PAGE_ASK:
        ask_reviewiq.render(table)
    elif page == PAGE_SELLER:
        business_intelligence.render(reviews, absa, table)


if __name__ == "__main__":
    main()
