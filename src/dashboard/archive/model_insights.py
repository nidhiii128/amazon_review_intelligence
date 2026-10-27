"""Page: Model Insights -- for the professor. Organised as three equal
pillars (NLP / Deep Learning / Advanced ML) so the project reads as a
balanced three-part system rather than an NLP project with extras."""
import re

import pandas as pd
import streamlit as st

from theme import FIG_DIR, METRICS_DIR, show_image_if_exists, show_text_if_exists
from widgets import render_pillars, pillar_headlines


def _metric(filename, pattern, cast=float):
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    m = re.search(pattern, path.read_text())
    return cast(m.group(1)) if m else None


def _sentiment_table():
    lstm_acc = _metric("lstm_results.txt", r"Accuracy: ([\d.]+)")
    lstm_f1 = _metric("lstm_results.txt", r"Macro F1: ([\d.]+)")
    rows = [
        ("VADER", "NLP", "81.6%", "0.519"),
        ("TF-IDF + Logistic Regression", "NLP / classical ML", "86.0%", "0.689"),
        ("TF-IDF + Linear SVM", "NLP / classical ML", "88.8%", "0.693"),
    ]
    if lstm_acc:
        rows.append(("BiLSTM + Attention", "Deep learning", f"{lstm_acc:.1%}", f"{lstm_f1:.3f}"))
    rows.append(("DistilBERT (fine-tuned)", "Deep learning", "89.9%", "0.702"))
    return pd.DataFrame(rows, columns=["Model", "Family", "Accuracy", "Macro F1"]).set_index("Model")


def _tab_overview():
    st.markdown("#### Sentiment model comparison (same test split for every model)")
    st.dataframe(_sentiment_table(), use_container_width=True)
    show_image_if_exists(FIG_DIR / "sentiment_model_comparison.png")
    st.caption("Sentiment labels are derived from star ratings (4-5 positive, 3 neutral, 1-2 negative); "
               "macro-F1 is the fairer metric because the data is ~80% positive.")


def _tab_nlp():
    st.markdown("#### Natural Language Processing")
    st.markdown(
        "- **VADER**: rule-based sentiment lexicon, the zero-training baseline\n"
        "- **TF-IDF + Logistic Regression / Linear SVM**: classical bag-of-words classifiers\n"
        "- **LDA**: unsupervised topic modelling (8 topics)\n"
        "- **ABSA**: aspect-level sentiment (battery, sound, camera, ...)\n"
        "- **Sentence embeddings + cosine similarity**: semantic search that feeds the RAG chatbot"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Aspect-level sentiment (ABSA)")
        show_image_if_exists(FIG_DIR / "aspect_sentiment_stacked.png")
    with c2:
        st.markdown("##### LDA topics")
        show_image_if_exists(FIG_DIR / "lda_top_words.png")
    show_text_if_exists(METRICS_DIR / "vader_results.txt", "VADER report")
    show_text_if_exists(METRICS_DIR / "tfidf_classical_results.txt", "TF-IDF models report")
    show_text_if_exists(METRICS_DIR / "absa_summary.txt", "ABSA summary")
    show_text_if_exists(METRICS_DIR / "lda_topics.txt", "LDA topics")


def _tab_dl():
    st.markdown("#### Deep Learning")
    st.markdown(
        "- **BiLSTM + Attention**: recurrent network trained from scratch in PyTorch (GPU), "
        "class-weighted loss, early stopping on validation macro-F1\n"
        "- **DistilBERT**: pre-trained transformer fine-tuned for 3-class sentiment\n"
        "- **FLAN-T5**: transformer used for grounded answer generation in the RAG chatbot\n\n"
        "The interactive version, with word clouds and a live classifier, is on the **🧠 Deep Learning Lab** page."
    )
    acc = _metric("lstm_results.txt", r"Accuracy: ([\d.]+)")
    f1 = _metric("lstm_results.txt", r"Macro F1: ([\d.]+)")
    b_acc = _metric("bert_results.txt", r"Accuracy: ([\d.]+)")
    b_f1 = _metric("bert_results.txt", r"Macro F1: ([\d.]+)")
    c1, c2, c3, c4 = st.columns(4)
    if acc:
        c1.metric("LSTM accuracy", f"{acc:.1%}")
        c2.metric("LSTM macro-F1", f"{f1:.3f}")
    if b_acc:
        c3.metric("DistilBERT accuracy", f"{b_acc:.1%}")
        c4.metric("DistilBERT macro-F1", f"{b_f1:.3f}")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("##### LSTM training curves")
        show_image_if_exists(FIG_DIR / "lstm_training_curves.png")
    with right:
        st.markdown("##### LSTM confusion matrix")
        show_image_if_exists(FIG_DIR / "lstm_confusion_matrix.png")
    show_text_if_exists(METRICS_DIR / "lstm_results.txt", "BiLSTM + Attention report")
    show_text_if_exists(METRICS_DIR / "bert_results.txt", "DistilBERT report")


def _tab_aml():
    st.markdown("#### Advanced Machine Learning")
    st.markdown(
        "- **XGBoost**: gradient-boosted trees predicting review helpfulness from engineered features, "
        "tuned with randomized hyperparameter search\n"
        "- **KMeans**: unsupervised clustering of review text (several clusters are generic praise, a known limitation)\n"
        "- **Isolation Forest**: unsupervised anomaly detection for potentially suspicious reviews "
        "(no labelled ground truth, so no accuracy claim)"
    )
    xgb = _metric("xgboost_helpfulness_results.txt", r"Accuracy: ([\d.]+)")
    xgb_f1 = _metric("xgboost_helpfulness_results.txt", r"F1 \(helpful\): ([\d.]+)")
    c1, c2, c3 = st.columns(3)
    if xgb:
        c1.metric("XGBoost accuracy", f"{xgb:.1%}")
        c2.metric("XGBoost F1 (helpful)", f"{xgb_f1:.3f}")
    flagged = _metric("suspicious_reviews_summary.txt", r"Potentially suspicious: ([\d,]+)", lambda x: int(x.replace(",", "")))
    if flagged:
        c3.metric("Reviews flagged (Isolation Forest)", f"{flagged:,}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### XGBoost feature importance")
        show_image_if_exists(FIG_DIR / "xgboost_feature_importance.png")
    with c2:
        st.markdown("##### KMeans clusters")
        show_image_if_exists(FIG_DIR / "kmeans_clusters.png")
    show_text_if_exists(METRICS_DIR / "xgboost_helpfulness_results.txt", "XGBoost report")
    show_text_if_exists(METRICS_DIR / "kmeans_clusters.txt", "KMeans clusters")
    show_text_if_exists(METRICS_DIR / "suspicious_reviews_summary.txt", "Isolation Forest summary")


def render():
    st.subheader("🧪 Model Insights")
    render_pillars(pillar_headlines())
    st.write("")
    tab_overview, tab_nlp, tab_dl, tab_aml, tab_rag = st.tabs(
        ["Overview", "🔤 NLP", "🧠 Deep Learning", "📊 Advanced ML", "💬 RAG"]
    )
    with tab_overview:
        _tab_overview()
    with tab_nlp:
        _tab_nlp()
    with tab_dl:
        _tab_dl()
    with tab_aml:
        _tab_aml()
    with tab_rag:
        st.markdown("#### Retrieval-Augmented Generation")
        st.markdown("Question -> sentence-embedding retrieval (top 5 reviews, cosine similarity, scoped to one "
                    "product) -> FLAN-T5 answer, with a similarity floor so unsupported questions get "
                    "\"not enough information\" instead of a guess.")
        show_text_if_exists(METRICS_DIR / "rag_chatbot_demo.txt", "RAG demo transcript")
