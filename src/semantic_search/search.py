"""
Milestone 5, step 2: semantic (meaning-based) review search.

Loads the embeddings built by build_embeddings.py and, for a query like
"battery problems", finds reviews that are similar in *meaning* even if
they never use the word "battery" -- e.g. "I have to charge it twice a
day." This is the core capability the RAG chatbot (Milestone 6) is built
on top of.

Since embeddings were saved L2-normalized, cosine similarity is just a
dot product -- no extra normalization needed at query time either.

Usage:
  python3 src/semantic_search/search.py                  # runs demo queries
  python3 src/semantic_search/search.py "is the camera good for low light"
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
EMBED_DIR = BASE_DIR / "models" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

DEMO_QUERIES = [
    "battery problems",
    "is the camera good for low light",
    "easy to set up",
    "stopped working after a few weeks",
    "great value for the price",
]


class SemanticSearcher:
    def __init__(self):
        embeddings_path = EMBED_DIR / "review_embeddings.npy"
        ids_path = EMBED_DIR / "review_ids.npy"
        if not embeddings_path.exists() or not ids_path.exists():
            raise SystemExit(
                "Embeddings not found -- run src/semantic_search/build_embeddings.py first."
            )
        print("Loading model + precomputed embeddings ...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = np.load(embeddings_path)
        self.review_ids = np.load(ids_path)
        df = pd.read_csv(DATA_PATH)
        self.df = df.set_index("review_id").loc[self.review_ids].reset_index()

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec  # cosine similarity (both sides normalized)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = self.df.iloc[top_idx].copy()
        results["similarity"] = scores[top_idx]
        return results[["review_id", "asin", "rating", "sentiment_label", "similarity", "clean_text"]]


def print_results(query, results):
    print(f"\n=== Query: \"{query}\" ===")
    for _, row in results.iterrows():
        snippet = row["clean_text"][:180].replace("\n", " ")
        print(f"  [{row['similarity']:.3f}] rating={row['rating']} "
              f"sentiment={row['sentiment_label']} asin={row['asin']}")
        print(f"      \"{snippet}...\"")


def main():
    searcher = SemanticSearcher()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = searcher.search(query, top_k=5)
        print_results(query, results)
    else:
        print("No query given -- running demo queries (try: "
              'python3 src/semantic_search/search.py "your question here")')
        for query in DEMO_QUERIES:
            results = searcher.search(query, top_k=3)
            print_results(query, results)


if __name__ == "__main__":
    main()
