"""
Unsupervised suspicious-review detection (Isolation Forest).

There is no labelled fake/genuine ground truth for this dataset, so this
is deliberately framed as anomaly detection over review-level features,
not a "fake review classifier" -- no accuracy claim is made anywhere in
its output. Flagged reviews are "potentially suspicious based on
automated pattern analysis," never asserted to be fake.

Features:
  - rating, review_length_words, capital_ratio, exclamation_count,
    verified_purchase_int, review_age_days (cheap text/metadata signals)
  - max_similarity_within_product: the highest cosine similarity between
    this review and any other review of the *same* parent_asin, reusing
    the Milestone 5 sentence embeddings (no new embedding pass). Computed
    per-product-group rather than a full N x N matrix (50k^2 pairs would
    be ~2.5B -- copy-paste spam is a same-product phenomenon anyway, so
    per-group blocks are both cheaper and the right scope).

Needs the embeddings from src/semantic_search/build_embeddings.py to
already exist.

Outputs:
  - data/processed/suspicious_reviews.csv   review_id, suspicion_score,
        suspicion_label, reasons
  - reports/metrics/suspicious_reviews_summary.txt
"""
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
EMBED_DIR = BASE_DIR / "models" / "embeddings"
OUT_PATH = BASE_DIR / "data" / "processed" / "suspicious_reviews.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CONTAMINATION = 0.10
NEAR_DUPLICATE_THRESHOLD = 0.95
SIMILAR_THRESHOLD = 0.85
SHORT_REVIEW_WORDS = 10
HIGH_CAPITAL_RATIO = 0.3
MANY_EXCLAMATIONS = 3

FEATURE_COLS = [
    "rating", "review_length_words", "capital_ratio", "exclamation_count",
    "verified_purchase_int", "review_age_days", "max_similarity_within_product",
]


def capital_ratio(text: str) -> float:
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    max_ts = df["timestamp"].max()
    df["review_age_days"] = (max_ts - df["timestamp"]).dt.days.fillna(0)
    df["verified_purchase_int"] = df["verified_purchase"].astype(int)
    df["exclamation_count"] = df["clean_text"].fillna("").astype(str).str.count(r"!")
    df["capital_ratio"] = df["clean_text"].fillna("").apply(capital_ratio)
    return df


def compute_max_similarity_within_product(df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    """For each review, the highest cosine similarity to any *other*
    review of the same parent_asin (0.0 for products with only 1 review).
    Embeddings are already L2-normalized, so dot product == cosine sim."""
    max_sim = np.zeros(len(df), dtype=np.float32)
    for _, idx in df.groupby("parent_asin").indices.items():
        if len(idx) < 2:
            continue
        block = embeddings[idx]
        sims = block @ block.T
        np.fill_diagonal(sims, -1.0)
        max_sim[idx] = sims.max(axis=1)
    return max_sim


def build_reasons(row: pd.Series) -> str:
    reasons = []
    if row["review_length_words"] < SHORT_REVIEW_WORDS:
        reasons.append("very short review")
    if row["max_similarity_within_product"] >= NEAR_DUPLICATE_THRESHOLD:
        reasons.append("near-duplicate text")
    elif row["max_similarity_within_product"] >= SIMILAR_THRESHOLD:
        reasons.append("highly similar to other reviews")
    if row["capital_ratio"] >= HIGH_CAPITAL_RATIO:
        reasons.append("unusually high capital-letter ratio")
    if row["exclamation_count"] >= MANY_EXCLAMATIONS:
        reasons.append("many exclamation marks")
    if not reasons:
        reasons.append("unusual combination of review features")
    return "; ".join(reasons)


def main():
    embeddings_path = EMBED_DIR / "review_embeddings.npy"
    ids_path = EMBED_DIR / "review_ids.npy"
    if not embeddings_path.exists() or not ids_path.exists():
        raise SystemExit("Embeddings not found -- run src/semantic_search/build_embeddings.py first.")

    embeddings = np.load(embeddings_path)
    review_ids = np.load(ids_path)
    df = pd.read_csv(DATA_PATH)
    df = df.set_index("review_id").loc[review_ids].reset_index()  # align row order to embeddings

    print("Building features ...")
    df = build_features(df)
    df["max_similarity_within_product"] = compute_max_similarity_within_product(df, embeddings)

    X = df[FEATURE_COLS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Fitting IsolationForest (contamination={CONTAMINATION}) on {len(df):,} reviews ...")
    model = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    model.fit(X_scaled)

    raw_scores = -model.score_samples(X_scaled)  # higher = more anomalous
    suspicion_score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

    df["suspicion_score"] = suspicion_score.round(4)
    df["suspicion_label"] = np.where(predictions == -1, "potentially_suspicious", "normal")
    df["reasons"] = df.apply(build_reasons, axis=1)

    out_df = df[["review_id", "suspicion_score", "suspicion_label", "reasons"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS}, MODELS_DIR / "isolation_forest_suspicious.joblib")

    n_flagged = (df["suspicion_label"] == "potentially_suspicious").sum()
    n_near_dup = (df["max_similarity_within_product"] >= NEAR_DUPLICATE_THRESHOLD).sum()
    summary = (
        "=== Unsupervised Suspicious Review Detection (Isolation Forest) ===\n"
        "No labelled fake/genuine ground truth exists for this dataset -- this is\n"
        "anomaly detection over review features, not a fake-review classifier, and\n"
        "no accuracy figure is reported for that reason.\n\n"
        f"Reviews analysed: {len(df):,}\n"
        f"Features: {', '.join(FEATURE_COLS)}\n"
        f"Contamination (target flagged rate): {CONTAMINATION:.0%}\n\n"
        f"Normal: {len(df) - n_flagged:,} ({(len(df) - n_flagged) / len(df):.1%})\n"
        f"Potentially suspicious: {n_flagged:,} ({n_flagged / len(df):.1%})\n"
        f"Near-duplicate reviews (similarity >= {NEAR_DUPLICATE_THRESHOLD}): {n_near_dup:,}\n"
    )
    print(summary)
    (METRICS_DIR / "suspicious_reviews_summary.txt").write_text(summary)
    print(f"Saved {OUT_PATH} and reports/metrics/suspicious_reviews_summary.txt")


if __name__ == "__main__":
    main()
