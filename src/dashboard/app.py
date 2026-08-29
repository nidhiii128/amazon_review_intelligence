"""
Milestone 7: the final dashboard, tying every earlier milestone together
into two views, exactly matching the project brief:

  Buyer view    -> "Should I buy this product?"
  Company view  -> "What should we improve?"

Styled as a lightweight e-commerce-style shell (dark header, tab nav,
product cards) using its own branding/colors -- not a clone of any real
site's logo or exact color values.

Run with:  streamlit run src/dashboard/app.py
(run from the amazon-review-intelligence/ folder, or Streamlit may not
find the sibling modules -- see the sys.path handling below either way)
"""
import sys
import pathlib
import pandas as pd
import streamlit as st

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "dashboard"))
sys.path.insert(0, str(BASE_DIR / "src" / "rag_chatbot"))

from data_helpers import (
    load_reviews, load_absa, get_top_products, get_product_overview,
    get_aspect_sentiment_for_asin, get_strengths_and_weaknesses,
    get_topic_mix_for_asin, get_corpus_top_complaints,
)

FIG_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR = BASE_DIR / "reports" / "metrics"

# Semantic status colors (validated for contrast/CVD-safety) -- used
# consistently everywhere positive/neutral/negative appears, in this app
# and in the matplotlib charts from earlier milestones.
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_CRITICAL = "#d03b3b"
COLOR_ACCENT = "#eb6834"
COLOR_ACCENT_DARK = "#d95926"
COLOR_NAVY = "#131a2b"
COLOR_NAVY_LIGHT = "#1f2a44"

st.set_page_config(page_title="ReviewIQ -- Product Review Intelligence", page_icon="🛍️", layout="wide")

CUSTOM_CSS = f"""
<style>
.stApp {{ background-color: #f9f9f7; }}
div.block-container {{ padding-top: 1rem; max-width: 1200px; }}

.topbar {{
    background: linear-gradient(180deg, {COLOR_NAVY} 0%, {COLOR_NAVY_LIGHT} 100%);
    margin: -1rem -1rem 1.2rem -1rem;
    padding: 20px 32px 16px 32px;
    border-radius: 0 0 14px 14px;
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;
}}
.topbar .brand {{ color: #ffffff; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }}
.topbar .brand span {{ color: {COLOR_ACCENT}; }}
.topbar .tagline {{ color: #b8c4d9; font-size: 13px; margin-top: 2px; }}
.topbar .stats {{ color: #e7ecf3; font-size: 13px; text-align: right; line-height: 1.5; }}
.topbar .stats b {{ color: {COLOR_ACCENT}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background-color: white; border-radius: 8px 8px 0 0; padding: 10px 24px;
    font-weight: 600; font-size: 15px; border: 1px solid #e4e7eb; border-bottom: none;
}}
.stTabs [aria-selected="true"] {{
    background-color: {COLOR_ACCENT} !important; color: white !important;
}}

div.stButton > button {{
    background-color: {COLOR_ACCENT}; color: white; border: none;
    border-radius: 6px; font-weight: 700; padding: 6px 18px;
}}
div.stButton > button:hover {{ background-color: {COLOR_ACCENT_DARK}; color: white; }}

[data-testid="stMetric"] {{
    background: white; border-radius: 10px; padding: 12px 10px;
    border: 1px solid #e4e7eb;
}}

.product-card {{
    background: white; border: 1px solid #e4e7eb; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px; height: 108px;
}}
.product-card .pname {{ font-weight: 700; font-size: 14px; color: {COLOR_NAVY}; }}
.product-card .pstars {{ color: {COLOR_ACCENT}; font-size: 14px; margin-top: 2px; }}
.product-card .pmeta {{ color: #6b7280; font-size: 12px; margin-top: 2px; }}
</style>
"""

STATUS_COLOR_MAP = [COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL]  # positive, neutral, negative


@st.cache_data
def _load_reviews():
    return load_reviews()


@st.cache_data
def _load_absa():
    return load_absa()


@st.cache_resource(show_spinner="Loading the local chatbot model (first time only, ~1-2 min) ...")
def _load_chatbot():
    from rag_pipeline import RAGChatbot
    return RAGChatbot()


def star_string(rating: float) -> str:
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def show_image_if_exists(path: pathlib.Path, caption: str = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not generated yet: run the script that produces `{path.name}`.")


def show_text_if_exists(path: pathlib.Path, expander_title: str):
    with st.expander(expander_title):
        if path.exists():
            st.text(path.read_text())
        else:
            st.info(f"Not generated yet: `{path.name}` doesn't exist. Run the corresponding script first.")


def render_aspect_chart(aspect_table: pd.DataFrame):
    if aspect_table.empty:
        st.info("Not enough aspect mentions for this product in the sample.")
        return
    chart_df = aspect_table[["positive", "neutral", "negative"]]
    st.bar_chart(chart_df, color=STATUS_COLOR_MAP)


def render_product_grid(top_products: pd.DataFrame, session_key: str, n: int = 8):
    """A clickable grid of product cards (ASIN + star rating + review
    count) instead of a plain dropdown -- selecting a card updates
    session_state[session_key] and reruns the app."""
    rows = top_products.head(n)
    cols = st.columns(4)
    for i, row in enumerate(rows.itertuples()):
        with cols[i % 4]:
            st.markdown(
                f"""<div class="product-card">
                    <div class="pname">{row.asin}</div>
                    <div class="pstars">{star_string(row.avg_rating)} {row.avg_rating}</div>
                    <div class="pmeta">{row.review_count:,} reviews</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("View", key=f"{session_key}_{row.asin}"):
                st.session_state[session_key] = row.asin


def product_picker(reviews: pd.DataFrame, label: str, key: str):
    """Card grid for browsing the top products, plus a search-style text
    box and a full dropdown as a fallback for finding any product."""
    top_products = get_top_products(reviews, n=200)
    session_key = f"selected_{key}"
    if session_key not in st.session_state:
        st.session_state[session_key] = top_products.iloc[0]["asin"]

    st.markdown(f"**{label}**")
    search = st.text_input("🔍 Search by ASIN", key=f"search_{key}", placeholder="Type part of an ASIN...")
    filtered = top_products[top_products["asin"].str.contains(search, case=False)] if search else top_products
    render_product_grid(filtered, session_key)

    options = top_products["asin"].tolist()
    current = st.session_state[session_key]
    idx = options.index(current) if current in options else 0
    chosen = st.selectbox(
        "Or pick from the full list", options, index=idx, key=f"select_{key}",
        format_func=lambda a: f"{a}  ({top_products.set_index('asin').loc[a, 'avg_rating']}★, "
                               f"{top_products.set_index('asin').loc[a, 'review_count']} reviews)",
    )
    st.session_state[session_key] = chosen
    return st.session_state[session_key]


def render_topbar(reviews: pd.DataFrame):
    st.markdown(
        f"""<div class="topbar">
            <div>
                <div class="brand">🛍️ Review<span>IQ</span></div>
                <div class="tagline">AI-Based Product Review Intelligence and Consumer Insight System</div>
            </div>
            <div class="stats">
                <b>{len(reviews):,}</b> reviews analyzed &nbsp;|&nbsp;
                <b>{reviews['asin'].nunique():,}</b> products &nbsp;|&nbsp;
                Electronics category sample
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_buyer_view(reviews, absa):
    st.subheader("Should I buy this?")
    st.caption(
        "Products are identified by ASIN (this dataset only pulled review-level "
        "fields, not product titles/metadata)."
    )

    asin = product_picker(reviews, "Choose a product", key="buyer_product")
    overview = get_product_overview(reviews, asin)

    st.markdown(f"### {asin}  {star_string(overview.get('avg_rating', 0))}")
    cols = st.columns(4)
    cols[0].metric("Reviews", overview.get("review_count", 0))
    cols[1].metric("Avg rating", f"{overview.get('avg_rating', 0)} / 5")
    cols[2].metric("% positive", f"{overview.get('positive_pct', 0)}%")
    cols[3].metric("% negative", f"{overview.get('negative_pct', 0)}%")

    aspect_table = get_aspect_sentiment_for_asin(absa, asin)
    sw = get_strengths_and_weaknesses(aspect_table)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Strengths")
        if sw["strengths"]:
            for s in sw["strengths"]:
                st.success(s.replace("_", " ").title())
        else:
            st.info("Not enough data.")
    with col_b:
        st.markdown("#### Weaknesses")
        if sw["weaknesses"]:
            for w in sw["weaknesses"]:
                st.error(w.replace("_", " ").title())
        else:
            st.info("Not enough data.")

    st.markdown("#### Aspect-level sentiment (% of mentions)")
    render_aspect_chart(aspect_table)

    topic_mix = get_topic_mix_for_asin(reviews, asin)
    if not topic_mix.empty:
        st.markdown("#### What reviews for this product talk about")
        st.bar_chart(topic_mix)

    st.markdown("---")
    st.markdown("#### Compare two products")
    col1, col2 = st.columns(2)
    with col1:
        asin_a = product_picker(reviews, "Product A", key="compare_a")
        st.write(f"**{asin_a}**", get_product_overview(reviews, asin_a))
        render_aspect_chart(get_aspect_sentiment_for_asin(absa, asin_a))
    with col2:
        asin_b = product_picker(reviews, "Product B", key="compare_b")
        st.write(f"**{asin_b}**", get_product_overview(reviews, asin_b))
        render_aspect_chart(get_aspect_sentiment_for_asin(absa, asin_b))

    st.markdown("---")
    st.markdown("#### 💬 Ask the AI chatbot about this product")
    st.caption(
        "Answers are generated by a local model grounded in the actual reviews "
        "shown below it -- not a general-knowledge guess."
    )
    question = st.text_input("Your question", placeholder="e.g. is the battery good?", key="buyer_question")
    if st.button("Ask", key="buyer_ask") and question:
        bot = _load_chatbot()
        with st.spinner("Retrieving reviews and generating an answer ..."):
            result = bot.ask(question, asin=asin)
        st.write("**Answer:**", result["answer"])
        st.write("**Supporting reviews:**")
        for _, row in result["supporting_reviews"].iterrows():
            st.caption(f"[similarity {row['similarity']:.2f}, rating {row['rating']}/5] {row['clean_text'][:200]}...")


def render_company_view(reviews, absa):
    st.subheader("What should we improve?")

    st.markdown("#### Top complaint areas across all products")
    complaints = get_corpus_top_complaints(absa, top_n=5)
    st.dataframe(complaints, use_container_width=True)

    st.markdown("#### Sentiment model comparison")
    show_image_if_exists(FIG_DIR / "sentiment_model_comparison.png")

    st.markdown("#### Aspect-level sentiment (all products)")
    show_image_if_exists(FIG_DIR / "aspect_sentiment_stacked.png")

    st.markdown("#### Topics discussed across reviews (LDA)")
    show_image_if_exists(FIG_DIR / "lda_top_words.png")

    st.markdown("#### Review helpfulness -- what predicts it")
    show_image_if_exists(FIG_DIR / "xgboost_feature_importance.png")

    st.markdown("#### Review clusters")
    show_image_if_exists(FIG_DIR / "kmeans_clusters.png")

    st.markdown("---")
    st.markdown("#### Per-product deep dive")
    asin = product_picker(reviews, "Choose a product to inspect", key="company_product")
    aspect_table = get_aspect_sentiment_for_asin(absa, asin)
    st.dataframe(aspect_table, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Full metric reports")
    show_text_if_exists(METRICS_DIR / "vader_results.txt", "VADER baseline")
    show_text_if_exists(METRICS_DIR / "tfidf_classical_results.txt", "TF-IDF classical models")
    show_text_if_exists(METRICS_DIR / "bert_results.txt", "DistilBERT fine-tuned")
    show_text_if_exists(METRICS_DIR / "absa_summary.txt", "Aspect-based sentiment summary")
    show_text_if_exists(METRICS_DIR / "lda_topics.txt", "LDA topics")
    show_text_if_exists(METRICS_DIR / "xgboost_helpfulness_results.txt", "XGBoost helpfulness prediction")
    show_text_if_exists(METRICS_DIR / "kmeans_clusters.txt", "KMeans clusters")


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    reviews = _load_reviews()
    absa = _load_absa()

    render_topbar(reviews)

    tab_buyer, tab_company = st.tabs(["🛍️  Buyer View", "🏢  Company View"])
    with tab_buyer:
        render_buyer_view(reviews, absa)
    with tab_company:
        render_company_view(reviews, absa)


if __name__ == "__main__":
    main()
