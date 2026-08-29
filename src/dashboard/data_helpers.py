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


def load_reviews() -> pd.DataFrame:
    path = TOPICS_PATH if TOPICS_PATH.exists() else REVIEWS_PATH
    return pd.read_csv(path)


def load_absa() -> pd.DataFrame:
    if not ABSA_PATH.exists():
        return pd.DataFrame(columns=["review_id", "asin", "aspect", "aspect_sentiment", "aspect_compound"])
    return pd.read_csv(ABSA_PATH)


def get_top_products(reviews: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Products with the most reviews, for the picker dropdown -- there's
    no product-title metadata in this dataset (only review-level fields
    were pulled), so products are identified by ASIN plus their stats."""
    agg = (
        reviews.groupby("asin")
        .agg(review_count=("review_id", "count"), avg_rating=("rating", "mean"))
        .sort_values("review_count", ascending=False)
        .head(n)
        .reset_index()
    )
    agg["avg_rating"] = agg["avg_rating"].round(2)
    return agg


def get_product_overview(reviews: pd.DataFrame, asin: str) -> dict:
    sub = reviews[reviews["asin"] == asin]
    if sub.empty:
        return {}
    return {
        "review_count": len(sub),
        "avg_rating": round(sub["rating"].mean(), 2),
        "verified_pct": round(sub["verified_purchase"].mean() * 100, 1),
        "positive_pct": round((sub["sentiment_label"] == "positive").mean() * 100, 1),
        "negative_pct": round((sub["sentiment_label"] == "negative").mean() * 100, 1),
    }


def get_aspect_sentiment_for_asin(absa: pd.DataFrame, asin: str) -> pd.DataFrame:
    """% positive/neutral/negative per aspect for one product. Falls back
    to an empty frame (caller should handle -- niche products may not have
    enough aspect mentions in the sample)."""
    sub = absa[absa["asin"] == asin]
    if sub.empty:
        return pd.DataFrame(columns=["aspect", "positive", "neutral", "negative", "mentions"])
    table = (pd.crosstab(sub["aspect"], sub["aspect_sentiment"], normalize="index") * 100).round(1)
    for col in ["positive", "neutral", "negative"]:
        if col not in table.columns:
            table[col] = 0.0
    table["mentions"] = sub["aspect"].value_counts()
    return table[["positive", "neutral", "negative", "mentions"]].sort_values("negative", ascending=False)


def get_strengths_and_weaknesses(aspect_table: pd.DataFrame, top_n: int = 2) -> dict:
    if aspect_table.empty:
        return {"strengths": [], "weaknesses": []}
    strengths = aspect_table.sort_values("positive", ascending=False).head(top_n).index.tolist()
    weaknesses = aspect_table.sort_values("negative", ascending=False).head(top_n).index.tolist()
    return {"strengths": strengths, "weaknesses": weaknesses}


def get_topic_mix_for_asin(reviews: pd.DataFrame, asin: str) -> pd.Series:
    if "dominant_topic_label" not in reviews.columns:
        return pd.Series(dtype=float)
    sub = reviews[reviews["asin"] == asin]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub["dominant_topic_label"].value_counts(normalize=True) * 100


def get_corpus_top_complaints(absa: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if absa.empty:
        return pd.DataFrame(columns=["aspect", "negative_pct", "mentions"])
    table = (pd.crosstab(absa["aspect"], absa["aspect_sentiment"], normalize="index") * 100).round(1)
    if "negative" not in table.columns:
        table["negative"] = 0.0
    table["mentions"] = absa["aspect"].value_counts()
    return table.sort_values("negative", ascending=False).head(top_n)[["negative", "mentions"]].rename(
        columns={"negative": "negative_pct"}
    )
