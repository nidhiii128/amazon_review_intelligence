"""
Pure data-loading and analysis functions for the dashboard, kept separate
from app.py (which wires these into the Streamlit UI) so they can be
tested and reused without needing a running Streamlit session.
"""
import pathlib
import pandas as pd
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

REVIEWS_PATH = DATA_DIR / "electronics_reviews_clean.csv"
TOPICS_PATH = DATA_DIR / "electronics_reviews_with_topics.csv"
ABSA_PATH = DATA_DIR / "absa_aspect_sentiments.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
SUSPICIOUS_PATH = DATA_DIR / "suspicious_reviews.csv"


def load_reviews() -> pd.DataFrame:
    path = TOPICS_PATH if TOPICS_PATH.exists() else REVIEWS_PATH
    return pd.read_csv(path)


def load_absa(reviews: pd.DataFrame = None) -> pd.DataFrame:
    """ABSA output is keyed by review_id/asin only (produced before
    metadata existed) -- merge in parent_asin from the reviews table at
    load time rather than rerunning aspect_sentiment.py."""
    if not ABSA_PATH.exists():
        return pd.DataFrame(columns=["review_id", "asin", "parent_asin", "aspect", "aspect_sentiment", "aspect_compound"])
    absa = pd.read_csv(ABSA_PATH)
    if reviews is not None and "parent_asin" not in absa.columns:
        absa = absa.merge(reviews[["review_id", "parent_asin"]], on="review_id", how="left")
    return absa


def load_products() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        return pd.DataFrame(columns=[
            "parent_asin", "product_name", "brand", "image_url",
            "main_category", "average_rating", "rating_number",
        ])
    return pd.read_csv(PRODUCTS_PATH)


def load_suspicious() -> pd.DataFrame:
    if not SUSPICIOUS_PATH.exists():
        return pd.DataFrame(columns=["review_id", "suspicion_score", "suspicion_label", "reasons"])
    return pd.read_csv(SUSPICIOUS_PATH)


def display_name(products: pd.DataFrame, parent_asin: str) -> str:
    """Product name if metadata was found for this parent_asin, otherwise
    a graceful fallback so the UI never shows a blank."""
    row = products[products["parent_asin"] == parent_asin]
    if row.empty or not str(row.iloc[0]["product_name"]).strip():
        return f"Product {parent_asin}"
    return row.iloc[0]["product_name"]


def get_top_products(reviews: pd.DataFrame, products: pd.DataFrame = None, n: int = 30) -> pd.DataFrame:
    """Products with the most reviews, for the picker/grid -- grouped by
    parent_asin (the product listing) rather than the variant-level asin,
    joined with product_name/brand/image_url when metadata was found."""
    agg = (
        reviews.groupby("parent_asin")
        .agg(review_count=("review_id", "count"), avg_rating=("rating", "mean"))
        .sort_values("review_count", ascending=False)
        .head(n)
        .reset_index()
    )
    agg["avg_rating"] = agg["avg_rating"].round(2)
    if products is not None and not products.empty:
        agg = agg.merge(
            products[["parent_asin", "product_name", "brand", "image_url"]],
            on="parent_asin", how="left",
        )
    if "product_name" not in agg.columns:
        agg["product_name"] = None
    agg["product_name"] = agg["product_name"].fillna(
        "Product " + agg["parent_asin"].astype(str)
    )
    return agg


def search_products_by_name(products: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return products
    return products[products["product_name"].str.contains(query, case=False, na=False)]


def get_product_overview(reviews: pd.DataFrame, parent_asin: str) -> dict:
    sub = reviews[reviews["parent_asin"] == parent_asin]
    if sub.empty:
        return {}
    return {
        "review_count": len(sub),
        "avg_rating": round(sub["rating"].mean(), 2),
        "verified_pct": round(sub["verified_purchase"].mean() * 100, 1),
        "positive_pct": round((sub["sentiment_label"] == "positive").mean() * 100, 1),
        "negative_pct": round((sub["sentiment_label"] == "negative").mean() * 100, 1),
    }


def get_aspect_sentiment_for_asin(absa: pd.DataFrame, parent_asin: str) -> pd.DataFrame:
    """% positive/neutral/negative per aspect for one product. Falls back
    to an empty frame (caller should handle -- niche products may not have
    enough aspect mentions in the sample)."""
    sub = absa[absa["parent_asin"] == parent_asin]
    if sub.empty:
        return pd.DataFrame(columns=["aspect", "positive", "neutral", "negative", "mentions"])
    table = (pd.crosstab(sub["aspect"], sub["aspect_sentiment"], normalize="index") * 100).round(1)
    for col in ["positive", "neutral", "negative"]:
        if col not in table.columns:
            table[col] = 0.0
    table["mentions"] = sub["aspect"].value_counts()
    return table[["positive", "neutral", "negative", "mentions"]].sort_values("negative", ascending=False)


def get_aspect_evidence(absa: pd.DataFrame, parent_asin: str, aspect: str, n: int = 10) -> pd.DataFrame:
    """The actual matched sentences behind one product's aspect score --
    reuses the `evidence` column aspect_sentiment.py already writes, so
    clicking an aspect in the UI is explainable without new extraction
    logic."""
    sub = absa[(absa["parent_asin"] == parent_asin) & (absa["aspect"] == aspect)]
    return sub.sort_values("aspect_compound").head(n)[
        ["review_id", "aspect_sentiment", "aspect_compound", "evidence"]
    ]


def get_strengths_and_weaknesses(aspect_table: pd.DataFrame, top_n: int = 2) -> dict:
    if aspect_table.empty:
        return {"strengths": [], "weaknesses": []}
    strengths = aspect_table.sort_values("positive", ascending=False).head(top_n).index.tolist()
    weaknesses = aspect_table.sort_values("negative", ascending=False).head(top_n).index.tolist()
    return {"strengths": strengths, "weaknesses": weaknesses}


def get_topic_mix_for_asin(reviews: pd.DataFrame, parent_asin: str) -> pd.Series:
    if "dominant_topic_label" not in reviews.columns:
        return pd.Series(dtype=float)
    sub = reviews[reviews["parent_asin"] == parent_asin]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub["dominant_topic_label"].value_counts(normalize=True) * 100


def get_corpus_top_complaints(absa: pd.DataFrame, top_n: int = 5, parent_asin: str = None) -> pd.DataFrame:
    if parent_asin:
        absa = absa[absa["parent_asin"] == parent_asin]
    if absa.empty:
        return pd.DataFrame(columns=["aspect", "negative_pct", "mentions"])
    table = (pd.crosstab(absa["aspect"], absa["aspect_sentiment"], normalize="index") * 100).round(1)
    if "negative" not in table.columns:
        table["negative"] = 0.0
    table["mentions"] = absa["aspect"].value_counts()
    return table.sort_values("negative", ascending=False).head(top_n)[["negative", "mentions"]].rename(
        columns={"negative": "negative_pct"}
    )


def aspect_status(row: pd.Series, mixed_band: float = 15.0) -> str:
    """positive / negative / mixed classification for the colored aspect
    list -- mixed when positive and negative are within `mixed_band`
    points of each other rather than one clearly dominating."""
    if abs(row["positive"] - row["negative"]) <= mixed_band:
        return "mixed"
    return "positive" if row["positive"] > row["negative"] else "negative"


def ai_summary_sentence(overview: dict, strengths: list, weaknesses: list) -> str:
    """Template-generated, not an LLM call -- built directly from computed
    stats so it can't invent anything not backed by the data."""
    if not overview:
        return "Not enough reviews for this product to summarize yet."
    pos = overview.get("positive_pct", 0)
    tone = "positive" if pos >= 60 else ("mixed" if pos >= 40 else "largely negative")
    parts = [f"Customers generally have a {tone} opinion ({pos}% positive reviews)."]
    if strengths:
        parts.append(f"{', '.join(s.replace('_', ' ').title() for s in strengths)} receive strong feedback.")
    if weaknesses:
        parts.append(f"{', '.join(w.replace('_', ' ').title() for w in weaknesses)} concerns appear repeatedly.")
    return " ".join(parts)


def get_comparison_table(reviews, absa, products, parent_asin_a, parent_asin_b) -> pd.DataFrame:
    """Side-by-side computed stats for two products -- everything here is
    a lookup/aggregation, no model inference, so the Compare page's copy
    can quote it directly without risk of invented numbers."""
    rows = []
    for label, pa in [("A", parent_asin_a), ("B", parent_asin_b)]:
        overview = get_product_overview(reviews, pa)
        aspects = get_aspect_sentiment_for_asin(absa, pa)
        top_negative = aspects.sort_values("negative", ascending=False).head(1)
        rows.append({
            "product": label,
            "name": display_name(products, pa),
            "rating": overview.get("avg_rating"),
            "reviews": overview.get("review_count"),
            "positive_pct": overview.get("positive_pct"),
            "negative_pct": overview.get("negative_pct"),
            "main_concern": top_negative.index[0].replace("_", " ").title() if not top_negative.empty else "n/a",
        })
    return pd.DataFrame(rows).set_index("product")


def ai_comparison_sentence(comparison: pd.DataFrame) -> str:
    """Template-generated from the computed comparison table above --
    never an LLM call, so it can't invent a comparison fact the data
    doesn't support."""
    if comparison.empty or len(comparison) < 2:
        return "Not enough data to compare these products yet."
    a, b = comparison.loc["A"], comparison.loc["B"]
    better = a["name"] if (a["positive_pct"] or 0) >= (b["positive_pct"] or 0) else b["name"]
    worse = b["name"] if better == a["name"] else a["name"]
    return (
        f"{better} receives stronger overall feedback ({max(a['positive_pct'] or 0, b['positive_pct'] or 0)}% positive) "
        f"than {worse}. Main concern for {a['name']}: {a['main_concern']}. "
        f"Main concern for {b['name']}: {b['main_concern']}."
    )


def get_top_helpful_predicted(df: pd.DataFrame, model, feature_cols: list, build_features_fn, n: int = 10) -> pd.DataFrame:
    """Reviews the trained XGBoost helpfulness model scores most likely to
    be helpful -- reuses the exact build_features() from
    helpfulness_xgboost.py so the deployed features match training."""
    featured = build_features_fn(df)
    proba = model.predict_proba(featured[feature_cols])[:, 1]
    featured = featured.assign(predicted_helpful_proba=proba)
    return featured.sort_values("predicted_helpful_proba", ascending=False).head(n)[
        ["review_id", "parent_asin", "rating", "clean_text", "predicted_helpful_proba"]
    ]


def get_suspicious_summary_for_asin(suspicious: pd.DataFrame, reviews: pd.DataFrame, parent_asin: str) -> dict:
    review_ids = reviews[reviews["parent_asin"] == parent_asin]["review_id"]
    sub = suspicious[suspicious["review_id"].isin(review_ids)]
    return {
        "total": len(sub),
        "normal": int((sub["suspicion_label"] == "normal").sum()),
        "suspicious": int((sub["suspicion_label"] == "potentially_suspicious").sum()),
        "flagged": sub[sub["suspicion_label"] == "potentially_suspicious"].sort_values(
            "suspicion_score", ascending=False
        ),
    }
