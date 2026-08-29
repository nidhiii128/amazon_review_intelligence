"""
Milestone 4 (optional secondary task): group similar reviews together with
KMeans clustering on the same TF-IDF vectors used by the Milestone 1
classical sentiment models (reuses models/tfidf_vectorizer.joblib -- no
retraining a vectorizer from scratch).

Only included because it adds a genuinely different lens than LDA topics
(bottom-up similarity grouping vs. word-distribution topics) and gives an
extra way to say "cluster 2 is mostly battery complaints" -- not added
just to tick an "another algorithm" box.
"""
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
for d in (METRICS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 6
N_TOP_TERMS = 10


def main():
    df = pd.read_csv(DATA_PATH)

    # Note: deliberately NOT reusing models/tfidf_vectorizer.joblib from the
    # sentiment milestone -- that vectorizer keeps stopwords/negations on
    # purpose (they carry sentiment signal), which drowns clustering in
    # generic "the/it/and" clusters. Clustering wants topical words instead,
    # so it gets its own English-stopword-filtered vectorizer, same idea as
    # the LDA topic model.
    vectorizer = TfidfVectorizer(
        max_features=5000, min_df=5, max_df=0.85,
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"].astype(str))
    feature_names = vectorizer.get_feature_names_out()
    joblib.dump(vectorizer, MODELS_DIR / "kmeans_tfidf_vectorizer.joblib")

    print(f"Clustering {X.shape[0]:,} reviews into {N_CLUSTERS} groups (MiniBatchKMeans) ...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, batch_size=1024)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    lines = ["=== Review Clustering (KMeans on TF-IDF) ===", ""]
    cluster_labels = {}
    for c in range(N_CLUSTERS):
        centroid = km.cluster_centers_[c]
        top_idx = centroid.argsort()[::-1][:N_TOP_TERMS]
        top_terms = [feature_names[i] for i in top_idx]
        cluster_labels[c] = ", ".join(top_terms[:3])
        count = (labels == c).sum()
        avg_rating = df.loc[df["cluster"] == c, "rating"].mean()
        lines.append(f"Cluster {c} ({count:,} reviews, avg rating {avg_rating:.2f}):")
        lines.append("  " + ", ".join(top_terms))
        lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)
    (METRICS_DIR / "kmeans_clusters.txt").write_text(summary_text)

    joblib.dump(km, MODELS_DIR / "kmeans_clusters.joblib")

    counts = df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(9, 5))
    x_labels = [f"C{c}\n{cluster_labels[c]}" for c in counts.index]
    plt.bar(x_labels, counts.values, color="#55a868")
    plt.ylabel("Number of reviews")
    plt.title("Review Clusters (KMeans on TF-IDF)")
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_clusters.png", dpi=150)
    plt.close()
    print(f"Saved chart to {FIG_DIR / 'kmeans_clusters.png'}")
    print(f"Saved summary to {METRICS_DIR / 'kmeans_clusters.txt'}")


if __name__ == "__main__":
    main()
