"""
Unsupervised suspicious-review detection (Isolation Forest).

There is no labelled fake/genuine ground truth for this dataset, so this
is deliberately framed as anomaly detection over review-level features,
not a "fake review classifier" -- no accuracy claim is made anywhere in
its output. Flagged reviews are "potentially suspicious based on
automated pattern analysis," never asserted to be fake.

Features are deliberately chosen for fraud-*directionality*, not just
statistical rarity -- an earlier version fed raw `rating` and raw
`review_length_words` directly into the model, and IsolationForest ended
up flagging long, detailed, negative reviews as "suspicious" simply
because they're numerically rare in a mostly-short-mostly-positive
dataset (verified by inspecting correlations: length corr +0.55, rating
corr -0.42 with the score -- backwards from what "suspicious" should
mean). Fixed by using directional/binary transforms and adding a real
fraud signal that was missing entirely:
  - is_short_text: binary (very short reviews are the classic low-effort
    pattern -- being long is never itself suspicious, so raw word count
    is not used).
  - capital_ratio, exclamation_count: emotional/spam-style text markers.
  - verified_purchase_int.
  - reviewer_review_count_log: log1p of how many reviews that same
    user_id has in this sample -- a small number of accounts have
    200-600+ reviews here, a well-established signal for bot/professional
    reviewer accounts, and it wasn't used at all before this fix.
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
PROLIFIC_REVIEWER_THRESHOLD = 20  # ~95th percentile of reviews-per-user in the sample

FEATURE_COLS = [
    "is_short_text", "capital_ratio", "exclamation_count",
    "verified_purchase_int", "max_similarity_within_product",
    "reviewer_review_count_log",
]


def capital_ratio(text: str) -> float:
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["verified_purchase_int"] = df["verified_purchase"].astype(int)
    df["exclamation_count"] = df["clean_text"].fillna("").astype(str).str.count(r"!")
    df["capital_ratio"] = df["clean_text"].fillna("").apply(capital_ratio)
    df["is_short_text"] = (df["review_length_words"] < SHORT_REVIEW_WORDS).astype(int)
    reviewer_counts = df["user_id"].value_counts()
    df["reviewer_review_count"] = df["user_id"].map(reviewer_counts)
    df["reviewer_review_count_log"] = np.log1p(df["reviewer_review_count"])
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
    if row["is_short_text"]:
        reasons.append("very short review")
    if row["max_similarity_within_product"] >= NEAR_DUPLICATE_THRESHOLD:
        reasons.append("near-duplicate text")
    elif row["max_similarity_within_product"] >= SIMILAR_THRESHOLD:
        reasons.append("highly similar to other reviews")
    if row["capital_ratio"] >= HIGH_CAPITAL_RATIO:
        reasons.append("unusually high capital-letter ratio")
    if row["exclamation_count"] >= MANY_EXCLAMATIONS:
        reasons.append("many exclamation marks")
    if row["reviewer_review_count"] >= PROLIFIC_REVIEWER_THRESHOLD:
        reasons.append(f"reviewer has posted {int(row['reviewer_review_count'])} reviews in this sample")
    if not row["verified_purchase_int"]:
        reasons.append("unverified purchase")
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
