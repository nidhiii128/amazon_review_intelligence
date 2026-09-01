"""
The final dashboard: a 9-page app tying every milestone together, matching
the finalized information architecture --
  Home, Explore Products, Product Insights, Compare Products,
  Ask ReviewIQ, Business Intelligence, Model Insights, About.

Run with:  streamlit run src/dashboard/app.py
(run from the amazon-review-intelligence/ folder, or Streamlit may not
find the sibling modules -- see the sys.path handling below either way)
"""
import sys
import pathlib

import streamlit as st

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "dashboard"))
sys.path.insert(0, str(BASE_DIR / "src" / "rag_chatbot"))
sys.path.insert(0, str(BASE_DIR / "src" / "advanced_ml"))

from data_helpers import load_reviews, load_absa, load_products, load_suspicious
from theme import inject_css, render_topbar
from pages import home, explore, insights, compare, ask_reviewiq, business_intelligence, model_insights, about

st.set_page_config(page_title="ReviewIQ -- Product Review Intelligence", page_icon="🧠", layout="wide")

PAGES = [
    "🏠 Home",
    "🔎 Explore Products",
    "📊 Product Insights",
    "⚖️ Compare Products",
    "🤖 Ask ReviewIQ",
    "📈 Business Intelligence",
    "🧪 Model Insights",
    "ℹ️ About",
]


@st.cache_data
def _load_reviews():
    return load_reviews()


@st.cache_data
def _load_absa(reviews):
    return load_absa(reviews)


@st.cache_data
def _load_products():
    return load_products()


@st.cache_data
def _load_suspicious():
    return load_suspicious()


def main():
    inject_css()
    reviews = _load_reviews()
    absa = _load_absa(reviews)
    products = _load_products()
    suspicious = _load_suspicious()

    render_topbar(reviews)

    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]

    page = st.sidebar.radio("Navigate", PAGES, index=PAGES.index(st.session_state["page"]))
    st.session_state["page"] = page

    if page == "🏠 Home":
        home.render(reviews)
    elif page == "🔎 Explore Products":
        explore.render(reviews, products)
    elif page == "📊 Product Insights":
        insights.render(reviews, absa, products, suspicious)
    elif page == "⚖️ Compare Products":
        compare.render(reviews, absa, products)
    elif page == "🤖 Ask ReviewIQ":
        ask_reviewiq.render(reviews, products)
    elif page == "📈 Business Intelligence":
        business_intelligence.render(reviews, absa, products)
    elif page == "🧪 Model Insights":
        model_insights.render()
    elif page == "ℹ️ About":
        about.render()


if __name__ == "__main__":
    main()
