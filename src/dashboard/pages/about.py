"""Page: About -- static disclaimers and dataset info."""
import streamlit as st


def render():
    st.subheader("ℹ️ About ReviewIQ")
    st.markdown(
        """
ReviewIQ turns unstructured Amazon product reviews (Electronics category)
into structured insight for two audiences: buyers deciding whether to
purchase, and companies deciding what to improve.

**Dataset:** [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(McAuley Lab), Electronics category, a ~50,000-review stratified working
sample, joined with product metadata (title, brand, image) via `parent_asin`.

**Pipeline:** NLP (VADER, TF-IDF + classical models, LDA, ABSA) → Deep
Learning (fine-tuned DistilBERT) → Advanced ML (XGBoost helpfulness,
KMeans clustering, Isolation Forest for suspicious-review detection) →
Semantic Search (Sentence-BERT embeddings + cosine similarity) → RAG
chatbot (retrieval + grounded generation).

**Important limitations, stated plainly:**
- Sentiment labels are derived from star ratings, not human-annotated.
- Several KMeans clusters are dominated by generic praise language rather
  than a distinct topic.
- Suspicious-review detection is **unsupervised** (Isolation Forest) --
  there is no labelled fake/genuine ground truth for this dataset, so no
  accuracy claim is made. Flagged reviews are described as "potentially
  suspicious based on automated pattern analysis," never as confirmed fake.
- The RAG chatbot only answers from retrieved reviews for the selected
  product; when reviews don't cover a question, it says so rather than
  inventing an answer.

No login or account is required to use this dashboard.
        """
    )
