"""
Milestone 6: RAG (Retrieval-Augmented Generation) chatbot.

Flow (matches the project brief exactly):
  question -> embed question -> retrieve top-K relevant reviews (cosine
  similarity over the Milestone 5 embeddings) -> feed question + those
  reviews to a small local LLM -> generate an answer -> show the answer
  AND the supporting reviews it was grounded in.

LLM: google/flan-t5-small, an instruction-tuned free/local model (no API
key, no internet after the first download) -- chosen because it's small
enough to run on a CPU-only laptop and follows "answer using only this
context" instructions reasonably well. Swap FLAN_MODEL_NAME for
"google/flan-t5-base" if you have more RAM/CPU and want better answers.

Needs the embeddings from src/semantic_search/build_embeddings.py to
already exist (run that first if you haven't). Needs `transformers` +
`torch` (already in requirements-deep-learning.txt).

Scoping is on parent_asin (the product listing) rather than the
variant-level asin, so color/size variants of the same listing are treated
as one product -- matches how the dashboard groups reviews and metadata.

Usage:
  python3 src/rag_chatbot/rag_pipeline.py                                   # demo questions
  python3 src/rag_chatbot/rag_pipeline.py "is the battery good"             # your own question
  python3 src/rag_chatbot/rag_pipeline.py "is this good for gaming" --parent-asin B00ZV9RDKK   # scoped to one product
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
EMBED_DIR = BASE_DIR / "models" / "embeddings"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
FLAN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 5
MAX_CONTEXT_CHARS_PER_REVIEW = 300

# Below this top-1 cosine similarity, retrieval is too weak to trust the
# generator's answer -- small local LLMs like flan-t5-small don't reliably
# follow "say you don't know" instructions on their own (verified: on an
# unsupported question the retrieved top-1 similarity was ~0.18-0.25 vs
# ~0.36-0.41 for a real but under-represented topic on the same product),
# so this is a hard guardrail rather than relying purely on the prompt.
MIN_SIMILARITY_THRESHOLD = 0.30
INSUFFICIENT_INFO_ANSWER = "The available customer reviews do not provide enough information to answer this question."

DEMO_QUESTIONS = [
    {"question": "What is the biggest complaint about battery life?", "parent_asin": None},
    {"question": "Is the camera good for low light photos?", "parent_asin": None},
    {"question": "Is this product good value for the price?", "parent_asin": None},
    # Product-scoped example (Fire TV Stick with Alexa Voice Remote --
    # Previous Generation, 175 reviews in the sample, avg rating 4.4) --
    # shows the buyer-facing "ask about THIS product" use case from the
    # project brief, not just corpus-wide search.
    {"question": "Is this good for cutting cable TV?", "parent_asin": "B075X8471B"},
]


class RAGChatbot:
    def __init__(self):
        embeddings_path = EMBED_DIR / "review_embeddings.npy"
        ids_path = EMBED_DIR / "review_ids.npy"
        if not embeddings_path.exists() or not ids_path.exists():
            raise SystemExit(
                "Embeddings not found -- run src/semantic_search/build_embeddings.py first."
            )

        print(f"Loading embedding model ({EMBED_MODEL_NAME}) ...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.embeddings = np.load(embeddings_path)
        self.review_ids = np.load(ids_path)

        df = pd.read_csv(DATA_PATH)
        self.df = df.set_index("review_id").loc[self.review_ids].reset_index()

        print(f"Loading local generation model ({FLAN_MODEL_NAME}, downloads on first run) ...")
        # Using the tokenizer/model classes directly (not the pipeline()
        # helper) -- pipeline task names like "text2text-generation" have
        # shifted across transformers versions, so this is more stable.
        self.gen_tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_NAME)

    def retrieve(self, query: str, parent_asin: str = None, top_k: int = TOP_K) -> pd.DataFrame:
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec

        if parent_asin:
            mask = (self.df["parent_asin"] == parent_asin).to_numpy()
            if mask.sum() == 0:
                print(f"(no reviews found for parent_asin={parent_asin}, searching the full corpus instead)")
            else:
                scores = np.where(mask, scores, -1.0)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = self.df.iloc[top_idx].copy()
        results["similarity"] = scores[top_idx]
        return results

    def generate_answer(self, question: str, context_reviews: pd.DataFrame) -> str:
        context_blocks = []
        for i, (_, row) in enumerate(context_reviews.iterrows(), 1):
            snippet = str(row["clean_text"])[:MAX_CONTEXT_CHARS_PER_REVIEW]
            context_blocks.append(f"Review {i} (rating {row['rating']}/5): {snippet}")
        context_text = "\n".join(context_blocks)

        prompt = (
            "You are summarizing customer reviews for a shopper. Using only the "
            "customer reviews below, write a 1-2 sentence answer to the question. "
            "Explain the reasoning briefly instead of answering with a single word. "
            "If the reviews don't say, answer \"The reviews don't mention this.\"\n\n"
            f"{context_text}\n\n"
            f"Question: {question}\n"
            "Answer in 1-2 full sentences:"
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.gen_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        return self.gen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def ask(self, question: str, parent_asin: str = None, top_k: int = TOP_K) -> dict:
        context_reviews = self.retrieve(question, parent_asin=parent_asin, top_k=top_k)
        if context_reviews.empty or context_reviews["similarity"].max() < MIN_SIMILARITY_THRESHOLD:
            answer = INSUFFICIENT_INFO_ANSWER
        else:
            answer = self.generate_answer(question, context_reviews)
        return {"question": question, "parent_asin": parent_asin, "answer": answer, "supporting_reviews": context_reviews}


def format_result(result: dict) -> str:
    lines = [f'Question: "{result["question"]}"' + (f" (product {result['parent_asin']})" if result["parent_asin"] else "")]
    lines.append(f"Answer: {result['answer']}")
    lines.append("Supporting reviews:")
    for _, row in result["supporting_reviews"].iterrows():
        snippet = str(row["clean_text"])[:150].replace("\n", " ")
        lines.append(f"  - [similarity {row['similarity']:.3f}, rating {row['rating']}/5] \"{snippet}...\"")
    return "\n".join(lines)


def main():
    bot = RAGChatbot()
    transcript = []

    args = sys.argv[1:]
    if args:
        parent_asin = None
        if "--parent-asin" in args:
            idx = args.index("--parent-asin")
            parent_asin = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        question = " ".join(args)
        result = bot.ask(question, parent_asin=parent_asin)
        text = format_result(result)
        print("\n" + text)
        transcript.append(text)
    else:
        print("No question given -- running demo questions (try: "
              'python3 src/rag_chatbot/rag_pipeline.py "your question" [--parent-asin PARENT_ASIN])')
        for item in DEMO_QUESTIONS:
            result = bot.ask(item["question"], parent_asin=item["parent_asin"])
            text = format_result(result)
            print("\n" + text)
            transcript.append(text)

    (METRICS_DIR / "rag_chatbot_demo.txt").write_text("\n\n".join(transcript))
    print(f"\nTranscript saved to {METRICS_DIR / 'rag_chatbot_demo.txt'}")


if __name__ == "__main__":
    main()
