"""Page 7: Model Insights -- for the professor. Verified metrics table +
full model inventory."""
import pandas as pd
import streamlit as st

from theme import METRICS_DIR, show_text_if_exists

# Hardcoded from the verified reports/metrics/*.txt files (not re-parsed
# at runtime) -- see bert_results.txt, tfidf_classical_results.txt,
# vader_results.txt.
SENTIMENT_COMPARISON = pd.DataFrame([
    {"Model": "VADER", "Accuracy": "81.6%", "Macro F1": "0.519"},
    {"Model": "Logistic Regression", "Accuracy": "86.0%", "Macro F1": "0.689"},
    {"Model": "Linear SVM", "Accuracy": "88.8%", "Macro F1": "0.693"},
    {"Model": "DistilBERT", "Accuracy": "89.9%", "Macro F1": "0.702"},
]).set_index("Model")

MODEL_INVENTORY = {
    "NLP": ["VADER", "TF-IDF", "Logistic Regression", "Linear SVM", "LDA", "ABSA"],
    "Deep Learning": ["Fine-tuned DistilBERT"],
    "Advanced ML": ["XGBoost (helpfulness)", "MiniBatchKMeans / KMeans (clustering)", "Isolation Forest (suspicious reviews)"],
    "Modern NLP / Retrieval": ["Sentence Transformers (all-MiniLM-L6-v2)", "Cosine Similarity"],
    "RAG": ["Retrieval (top-5, cosine similarity)", "Generation (FLAN-T5)"],
}


def render():
    st.subheader("🧪 Model Insights")

    st.markdown("#### Sentiment Model Comparison")
    st.dataframe(SENTIMENT_COMPARISON, use_container_width=True)

    st.markdown("#### Model Inventory")
    for category, models in MODEL_INVENTORY.items():
        st.markdown(f"**{category}**")
        st.markdown("\n".join(f"- {m}" for m in models))

    st.markdown("---")
    st.markdown("#### Full evaluation reports")
    show_text_if_exists(METRICS_DIR / "vader_results.txt", "VADER baseline")
    show_text_if_exists(METRICS_DIR / "tfidf_classical_results.txt", "TF-IDF classical models")
    show_text_if_exists(METRICS_DIR / "bert_results.txt", "DistilBERT fine-tuned")
    show_text_if_exists(METRICS_DIR / "absa_summary.txt", "Aspect-based sentiment summary")
    show_text_if_exists(METRICS_DIR / "lda_topics.txt", "LDA topics")
    show_text_if_exists(METRICS_DIR / "xgboost_helpfulness_results.txt", "XGBoost helpfulness prediction")
    show_text_if_exists(METRICS_DIR / "kmeans_clusters.txt", "KMeans clusters")
    show_text_if_exists(METRICS_DIR / "rag_chatbot_demo.txt", "RAG chatbot demo transcript")
