"""
Data preparation for the AI-Based Product Review Intelligence project.

Reads the raw Amazon Reviews 2023 (Electronics) JSONL sample, cleans it,
deduplicates it, derives a rating-based sentiment label (used later as a
weak-supervision target for the classical/DL sentiment models), and writes
a processed CSV capped at ~50,000 rows.
"""
import json
import re
import html
import pathlib
import pandas as pd
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "electronics_raw_sample.jsonl"
OUT_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
TARGET_SAMPLE_SIZE = 50000

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<.*?>")
WHITESPACE_RE = re.compile(r"\s+")


def load_raw(path: pathlib.Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append({
                "rating": obj.get("rating"),
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
                "asin": obj.get("asin"),
                "parent_asin": obj.get("parent_asin"),
                "user_id": obj.get("user_id"),
                "timestamp": obj.get("timestamp"),
                "helpful_vote": obj.get("helpful_vote", 0),
                "verified_purchase": obj.get("verified_purchase", False),
            })
    return pd.DataFrame.from_records(records)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def rating_to_label(rating: float) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw(RAW_PATH)
    print(f"Raw rows loaded: {len(df):,}")

    # Drop rows without usable text or rating
    df = df.dropna(subset=["rating"])
    df["text"] = df["text"].fillna("")
    df["title"] = df["title"].fillna("")
    df["review_full_text"] = (df["title"].astype(str) + ". " + df["text"].astype(str)).str.strip()
    df["clean_text"] = df["review_full_text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.split().str.len() >= 3]
    print(f"Dropped {before - len(df):,} rows with fewer than 3 words after cleaning")

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "asin", "clean_text"])
    print(f"Dropped {before - len(df):,} duplicate rows")

    df["rating"] = df["rating"].astype(float)
    df["sentiment_label"] = df["rating"].apply(rating_to_label)
    df["review_length_words"] = df["clean_text"].str.split().str.len()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].astype(bool)

    # Cap / sample to target size, stratified by sentiment label so all
    # three classes are represented in reasonable proportion.
    if len(df) > TARGET_SAMPLE_SIZE:
        frac = TARGET_SAMPLE_SIZE / len(df)
        sampled_parts = [
            group.sample(frac=frac, random_state=42)
            for _, group in df.groupby("sentiment_label")
        ]
        df = pd.concat(sampled_parts, ignore_index=True)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["review_id"] = np.arange(len(df))

    cols = [
        "review_id", "asin", "parent_asin", "user_id", "timestamp",
        "rating", "sentiment_label", "title", "text", "clean_text",
        "review_length_words", "helpful_vote", "verified_purchase",
    ]
    df = df[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nFinal processed dataset: {len(df):,} rows")
    print("Sentiment label distribution:")
    print(df["sentiment_label"].value_counts())
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
