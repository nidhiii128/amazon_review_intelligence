"""
Topic modeling with Latent Dirichlet Allocation (LDA).

Discovers what customers frequently discuss (battery, camera, shipping,
etc.) without any manual labeling. Uses sklearn's built-in English stopword
list, so it has no extra data-download dependency.

Outputs:
  - reports/metrics/lda_topics.txt         top words per topic
  - reports/figures/lda_top_words.png      bar chart of top words per topic
  - data/processed/electronics_reviews_with_topics.csv
        original processed data + dominant_topic (id) + dominant_topic_label
"""
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
OUT_DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_with_topics.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_TOPICS = 8
N_TOP_WORDS = 12


def main():
    df = pd.read_csv(DATA_PATH)
    texts = df["clean_text"].astype(str)

    print("Vectorizing text (CountVectorizer, English stopwords, unigrams+bigrams) ...")
    vectorizer = CountVectorizer(
        max_features=3000, min_df=5, max_df=0.85,
        stop_words="english", ngram_range=(1, 2),
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Fit LDA on a subsample for speed, then transform the full matrix.
    # (fit is the expensive part; transform on an already-fit model is cheap)
    fit_sample_size = min(8000, doc_term_matrix.shape[0])
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(doc_term_matrix.shape[0], size=fit_sample_size, replace=False)

    print(f"Fitting LDA with {N_TOPICS} topics on a {fit_sample_size:,}-doc subsample ...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=42, learning_method="online",
        max_iter=6, batch_size=1024, n_jobs=-1,
    )
    lda.fit(doc_term_matrix[fit_idx])

    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lda, MODELS_DIR / "lda_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "lda_vectorizer.joblib")
    print("Saved LDA model + vectorizer to models/")

    print("Transforming full dataset ...")
    doc_topic = lda.transform(doc_term_matrix)

    topic_lines = []
    topic_labels = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:N_TOP_WORDS]
        top_words = [feature_names[i] for i in top_indices]
        label = ", ".join(top_words[:3])
        topic_labels[topic_idx] = label
        topic_lines.append(f"Topic {topic_idx} ({label}):\n  " + ", ".join(top_words))

    topics_text = "\n\n".join(topic_lines)
    print(topics_text)
    (METRICS_DIR / "lda_topics.txt").write_text(topics_text)

    df["dominant_topic"] = doc_topic.argmax(axis=1)
    df["dominant_topic_label"] = df["dominant_topic"].map(topic_labels)

    print("\nReview counts per topic:")
    topic_counts = df["dominant_topic_label"].value_counts()
    print(topic_counts.to_string())

    df.to_csv(OUT_DATA_PATH, index=False)
    print(f"\nSaved topic-augmented dataset to {OUT_DATA_PATH}")

    # Bar chart: top words per topic (grid of subplots)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:8]
        top_words = [feature_names[i] for i in top_indices]
        weights = topic[top_indices]
        ax = axes[topic_idx]
        ax.barh(range(len(top_words)), weights, color="#4c72b0")
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Topic {topic_idx}", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lda_top_words.png", dpi=150)
    plt.close()
    print(f"Saved topic word chart to {FIG_DIR / 'lda_top_words.png'}")


if __name__ == "__main__":
    main()
