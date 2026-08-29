"""
Milestone 5, step 1: encode every review into a sentence embedding so
search can work by meaning instead of exact keyword overlap.

Model: all-MiniLM-L6-v2 (sentence-transformers). Chosen because it's small
(~80MB), fast enough for CPU, and a standard, well-tested choice for
semantic search / retrieval -- the same embeddings this script produces
are reused as-is by the Milestone 6 RAG chatbot's retrieval step, so this
only needs to run once.

Needs `sentence-transformers` (see requirements-deep-learning.txt) and
downloads the pretrained model from Hugging Face on first run (needs a
normal internet connection for that one-time download, same as the BERT
step).

Outputs:
  - models/embeddings/review_embeddings.npy   (N x 384 float32 matrix)
  - models/embeddings/review_ids.npy          (N,) review_id, same row order
"""
import time
import pathlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
EMBED_DIR = BASE_DIR / "models" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    texts = df["clean_text"].astype(str).tolist()
    review_ids = df["review_id"].to_numpy()

    print(f"Loading embedding model: {MODEL_NAME} (downloads on first run) ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts):,} reviews (batch size {BATCH_SIZE}) -- "
          f"this can take several minutes on CPU-only hardware ...")
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    np.save(EMBED_DIR / "review_embeddings.npy", embeddings.astype(np.float32))
    np.save(EMBED_DIR / "review_ids.npy", review_ids)

    elapsed = time.time() - t0
    print(f"\nSaved {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]} "
          f"to {EMBED_DIR} in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
