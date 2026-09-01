"""Page 6: Business Intelligence -- corpus-wide or per-product analysis
scope, 4 tabs: Overview, Complaint & Aspect Intelligence, Topics &
Discussions, Review Quality & Trust."""
import pathlib

import joblib
import streamlit as st

from data_helpers import (
    display_name, get_product_overview, get_corpus_top_complaints,
    get_top_products, get_top_helpful_predicted, load_suspicious,
)
from theme import FIG_DIR, METRICS_DIR, show_image_if_exists, show_text_if_exists

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


def render(reviews, absa, products):
    st.subheader("📈 Business Intelligence")

    top_products = get_top_products(reviews, products, n=500)
    scope = st.selectbox("Analysis Scope", ["All Products", "Specific Product"])
    scoped_parent_asin = None
    if scope == "Specific Product":
        options = top_products["parent_asin"].tolist()
        lookup = top_products.set_index("parent_asin")
        scoped_parent_asin = st.selectbox(
            "Product", options,
            format_func=lambda pa: lookup.loc[pa, "product_name"][:70],
        )

    tab_overview, tab_complaints, tab_topics, tab_trust = st.tabs(
        ["Overview", "🔥 Complaint & Aspect Intelligence", "🗂️ Topics & Discussions", "🛡️ Review Quality & Trust"]
    )

    with tab_overview:
        if scope == "All Products":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total reviews", f"{len(reviews):,}")
            c2.metric("Total products", f"{reviews['parent_asin'].nunique():,}")
            c3.metric("Average rating", f"{reviews['rating'].mean():.2f}")
            pos_pct = (reviews["sentiment_label"] == "positive").mean() * 100
            c4.metric("% Positive", f"{pos_pct:.1f}%")
            st.markdown("##### Sentiment distribution")
            st.bar_chart(reviews["sentiment_label"].value_counts())
        else:
            overview = get_product_overview(reviews, scoped_parent_asin)
            st.markdown(f"### {display_name(products, scoped_parent_asin)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rating", f"{overview.get('avg_rating', 0)} / 5")
            c2.metric("Reviews", overview.get("review_count", 0))
            c3.metric("% Positive", f"{overview.get('positive_pct', 0)}%")

    with tab_complaints:
        st.markdown("##### 🔥 Top Complaint Areas")
        complaints = get_corpus_top_complaints(
            absa, top_n=10, parent_asin=scoped_parent_asin if scope == "Specific Product" else None,
        )
        for rank, (aspect, r) in enumerate(complaints.iterrows(), 1):
            st.markdown(f"**{rank}. {aspect.replace('_', ' ').title()}** — "
                        f"Negative feedback: {r['negative_pct']}% ({int(r['mentions'])} mentions)")

    with tab_topics:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🗂️ LDA Topics — what are customers talking about?")
            show_image_if_exists(FIG_DIR / "lda_top_words.png")
            show_text_if_exists(METRICS_DIR / "lda_topics.txt", "Full LDA topic report")
        with col_b:
            st.markdown("##### 🧩 Review Clusters — how are similar reviews grouped?")
            show_image_if_exists(FIG_DIR / "kmeans_clusters.png")
            st.caption(
                "Several clusters are dominated by short, generic praise language "
                "(e.g. \"great product\", \"five stars\") rather than a distinct topic -- "
                "a known limitation of KMeans on short, ratings-skewed review text, "
                "not a bug. LDA separates topics more cleanly on this data."
            )
            show_text_if_exists(METRICS_DIR / "kmeans_clusters.txt", "Full KMeans cluster report")

    with tab_trust:
        st.markdown("##### 👍 Helpfulness")
        show_image_if_exists(FIG_DIR / "xgboost_feature_importance.png")
        model_path = MODELS_DIR / "xgboost_helpfulness.joblib"
        if model_path.exists():
            from helpfulness_xgboost import build_features, FEATURE_COLS
            model = joblib.load(model_path)
            top_helpful = get_top_helpful_predicted(reviews, model, FEATURE_COLS, build_features, n=10)
            st.markdown("Most informative reviews (highest predicted helpfulness):")
            st.dataframe(
                top_helpful[["rating", "predicted_helpful_proba", "clean_text"]],
                use_container_width=True,
            )
        else:
            st.info("XGBoost model not found -- run src/advanced_ml/helpfulness_xgboost.py first.")
        show_text_if_exists(METRICS_DIR / "xgboost_helpfulness_results.txt", "XGBoost helpfulness results")

        st.markdown("---")
        st.markdown("##### 🛡️ Suspicious Reviews")
        suspicious = load_suspicious()
        if suspicious.empty:
            st.info("Not generated yet -- run src/advanced_ml/suspicious_reviews.py.")
        else:
            flagged = suspicious[suspicious["suspicion_label"] == "potentially_suspicious"]
            c1, c2 = st.columns(2)
            c1.metric("Reviews analysed", f"{len(suspicious):,}")
            c2.metric("Flagged as potentially suspicious", f"{len(flagged):,}")
            st.caption(
                "Unsupervised detection (Isolation Forest) -- there is no labelled "
                "fake/genuine ground truth for this dataset, so this is presented as "
                "anomaly detection, not a fake-review accuracy claim."
            )
            st.dataframe(
                flagged.sort_values("suspicion_score", ascending=False).head(20),
                use_container_width=True,
            )
