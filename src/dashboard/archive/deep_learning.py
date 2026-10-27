"""Page: Deep Learning Lab -- the BiLSTM+Attention sentiment classifier
(architecture, training, metrics), the model comparison, word clouds, and a
live classifier that runs the LSTM, DistilBERT and the TF-IDF SVM side by
side on any review text."""
import html
import json
import pathlib
import re

import joblib
import pandas as pd
import streamlit as st

from theme import FIG_DIR, METRICS_DIR, show_image_if_exists

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
LABELS = ["negative", "neutral", "positive"]
LABEL_EMOJI = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}

SAMPLES = {
    "Mixed (aspect-level)": "The battery is terrible and dies in two hours, but the sound quality is amazing.",
    "Clearly positive": "Absolutely love this! Works perfectly, easy to set up and great value for the price.",
    "Clearly negative": "Stopped working after a week. Complete waste of money, I want a refund.",
    "Neutral / lukewarm": "It's okay. Does the job but nothing special, and the build feels a bit cheap.",
}

PIPELINE = [
    ("📝", "Review text"), ("✂️", "Tokenizer<br>15k vocab"), ("🔢", "Embedding<br>128-d"),
    ("🔁", "BiLSTM x2<br>128 hidden"), ("🎯", "Attention<br>pooling"), ("🧮", "Dense +<br>Softmax"),
    ("🏷️", "pos / neu / neg"),
]


@st.cache_resource(show_spinner="Loading the LSTM model ...")
def _load_lstm():
    import torch
    from lstm_model import load_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, cfg = load_checkpoint(MODELS_DIR / "lstm_sentiment.pt", device)
    return model, vocab, cfg, device


@st.cache_resource(show_spinner="Loading DistilBERT (first time only) ...")
def _load_bert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = MODELS_DIR / "bert_sentiment"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return tok, model, device


@st.cache_resource
def _load_svm():
    return joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib"), joblib.load(MODELS_DIR / "tfidf_svm.joblib")


def _read_lstm_results():
    path = METRICS_DIR / "lstm_results.txt"
    if not path.exists():
        return None
    text = path.read_text()
    g = lambda pat, cast=str: cast(re.search(pat, text).group(1))
    rows = []
    for label in LABELS:
        m = re.search(rf"{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
        rows.append({"Class": label.title(), "Precision": float(m[1]), "Recall": float(m[2]),
                     "F1": float(m[3]), "Support": int(m[4])})
    return {
        "accuracy": g(r"Accuracy: ([\d.]+)", float), "macro_f1": g(r"Macro F1: ([\d.]+)", float),
        "params": g(r"Trainable parameters: ([\d,]+)"), "epochs": g(r"Epochs run: (\d+)"),
        "best_epoch": g(r"best epoch by val macro-F1: (\d+)"), "minutes": g(r"Training time: ([\d.]+) minutes"),
        "per_class": pd.DataFrame(rows).set_index("Class"), "text": text,
    }


def _probs_bar(probs):
    df = pd.DataFrame({"probability": probs}, index=[f"{LABEL_EMOJI[l]} {l}" for l in LABELS])
    st.bar_chart(df, height=170, color="#eb6834")


def _attention_html(tokens_weights):
    peak = max(w for _, w in tokens_weights) or 1.0
    spans = []
    for tok, w in tokens_weights:
        alpha = min(1.0, w / peak) * 0.85
        spans.append(
            f'<span style="background:rgba(235,104,52,{alpha:.2f});padding:2px 4px;border-radius:4px;'
            f'margin:1px;display:inline-block;">{html.escape(tok)}</span>'
        )
    return f'<div style="line-height:2.1;font-size:15px;">{"".join(spans)}</div>'


def _tab_overview(res):
    st.markdown("#### BiLSTM + Attention sentiment classifier")
    st.caption("A recurrent neural network trained from scratch (PyTorch) to classify each review as "
               "positive, neutral or negative; the deep-learning counterpart to the TF-IDF models.")
    steps = []
    for i, (icon, label) in enumerate(PIPELINE):
        if i:
            steps.append('<div class="pipeline-arrow">→</div>')
        steps.append(f'<div class="pipeline-step" style="min-width:90px;padding:12px 8px;">'
                     f'<div class="picon">{icon}</div><div class="plabel">{label}</div></div>')
    st.markdown(f'<div class="pipeline-row">{"".join(steps)}</div>', unsafe_allow_html=True)
    st.write("")
    if res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{res['accuracy']:.1%}")
        c2.metric("Macro F1", f"{res['macro_f1']:.3f}")
        c3.metric("Parameters", res["params"])
        c4.metric("Training time", f"{res['minutes']} min (GPU)")
    st.markdown(
        "**Why these choices**\n"
        "- **Bidirectional LSTM**: reads each review forwards and backwards, so negation and context "
        "(\"not good\", \"good *until* it broke\") are captured, which bag-of-words models cannot do.\n"
        "- **Attention pooling**: learns which words matter most for the label instead of relying on the "
        "last hidden state; you can see this in the *Try it live* tab.\n"
        "- **Class-weighted loss + early stopping on validation macro-F1**: the data is 80% positive, so "
        "plain accuracy would reward ignoring the neutral and negative classes.\n"
        "- **Same test split** as the TF-IDF models and DistilBERT, so every number here is a like-for-like comparison."
    )


def _tab_metrics(res):
    if not res:
        st.info("Not generated yet. Run `src/deep_learning/lstm_sentiment.py`.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{res['accuracy']:.3f}")
    c2.metric("Macro F1", f"{res['macro_f1']:.3f}")
    c3.metric("Best epoch", f"{res['best_epoch']} of {res['epochs']}")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("##### Training curves")
        show_image_if_exists(FIG_DIR / "lstm_training_curves.png")
    with right:
        st.markdown("##### Confusion matrix (test set)")
        show_image_if_exists(FIG_DIR / "lstm_confusion_matrix.png")
    st.markdown("##### Per-class performance (test set, 10,000 reviews)")
    st.dataframe(res["per_class"].style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                 use_container_width=True)
    st.caption("Neutral is the hardest class for every model (it is only 7% of the data and sits between the "
               "other two); the LSTM's class weighting is what lifts its neutral recall.")
    with st.expander("Full evaluation report"):
        st.text(res["text"])


def _tab_comparison(res):
    rows = [
        ("VADER", "NLP (rule-based)", "81.6%", "0.519", "no training"),
        ("TF-IDF + Logistic Regression", "NLP + classical ML", "86.0%", "0.689", "40k (balanced weights)"),
        ("TF-IDF + Linear SVM", "NLP + classical ML", "88.8%", "0.693", "40k (balanced weights)"),
    ]
    if res:
        rows.append(("BiLSTM + Attention", "Deep learning", f"{res['accuracy']:.1%}", f"{res['macro_f1']:.3f}", "36k (sqrt class weights)"))
    rows.append(("DistilBERT (fine-tuned)", "Deep learning (transformer)", "89.9%", "0.702", "6k only (CPU run)"))
    df = pd.DataFrame(rows, columns=["Model", "Family", "Accuracy", "Macro F1", "Training data"]).set_index("Model")
    st.dataframe(df, use_container_width=True)
    show_image_if_exists(FIG_DIR / "sentiment_model_comparison.png")
    st.caption(
        "Reading this fairly: the two deep-learning models are the strongest overall: DistilBERT on accuracy, "
        "the LSTM on macro-F1 (best on the minority classes). Caveats: the TF-IDF models and the LSTM all use class "
        "weighting (which helps macro-F1), and DistilBERT saw only 6,000 training reviews because it was "
        "fine-tuned on CPU, so it is under-trained relative to the LSTM."
    )


def _tab_wordclouds():
    st.markdown("#### What each sentiment sounds like")
    st.caption("Word size = how characteristic a word is of that class (frequency weighted by how "
               "over-represented it is versus all reviews), so generic words don't dominate every cloud.")
    show_image_if_exists(FIG_DIR / "wordcloud_all_sentiments.png")
    c1, c2, c3 = st.columns(3)
    for col, label in zip((c1, c2, c3), LABELS):
        with col:
            st.markdown(f"**{LABEL_EMOJI[label]} {label.title()}**")
            show_image_if_exists(FIG_DIR / f"wordcloud_{label}.png")


def _tab_live():
    st.markdown("#### Try it live")
    st.caption("Type any review and see how the deep-learning models classify it, side by side with the classical SVM.")
    choice = st.selectbox("Example reviews", list(SAMPLES) + ["(write your own)"])
    default = SAMPLES.get(choice, "")
    text = st.text_area("Review text", value=default, height=110, key=f"live_text_{choice}",
                        placeholder="e.g. The battery is terrible but the sound is amazing.")
    if st.button("Classify review") and text.strip():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🔁 BiLSTM + Attention")
            if (MODELS_DIR / "lstm_sentiment.pt").exists():
                from lstm_model import predict
                model, vocab, cfg, device = _load_lstm()
                probs, attn = predict(model, vocab, cfg, [text], device, with_attention=True)
                label = LABELS[int(probs[0].argmax())]
                st.success(f"{LABEL_EMOJI[label]} **{label.title()}** ({probs[0].max():.0%})")
                _probs_bar(probs[0])
            else:
                attn = None
                st.info("Run `src/deep_learning/lstm_sentiment.py` first.")
        with c2:
            st.markdown("##### 🤖 DistilBERT")
            if (MODELS_DIR / "bert_sentiment").exists():
                import torch
                tok, bert, device = _load_bert()
                with torch.no_grad():
                    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                    bprobs = torch.softmax(bert(**enc).logits, dim=1)[0].cpu().numpy()
                label = LABELS[int(bprobs.argmax())]
                st.success(f"{LABEL_EMOJI[label]} **{label.title()}** ({bprobs.max():.0%})")
                _probs_bar(bprobs)
            else:
                st.info("Fine-tuned DistilBERT not found.")
        with c3:
            st.markdown("##### 📐 TF-IDF + Linear SVM")
            vec, svm = _load_svm()
            label = svm.predict(vec.transform([text]))[0]
            st.success(f"{LABEL_EMOJI[label]} **{label.title()}**")
            st.caption("A linear SVM gives a label but no calibrated probabilities.")
        if attn:
            st.markdown("##### 🎯 What the LSTM paid attention to")
            st.markdown(_attention_html(attn[0]), unsafe_allow_html=True)
            st.caption("Darker = higher attention weight. These weights are learned during training, "
                       "not hand-written rules.")


def render():
    st.subheader("🧠 Deep Learning Lab")
    res = _read_lstm_results()
    tabs = st.tabs(["Overview", "📈 Training & Metrics", "⚖️ Model Comparison", "☁️ Word Clouds", "⚡ Try it live"])
    with tabs[0]:
        _tab_overview(res)
    with tabs[1]:
        _tab_metrics(res)
    with tabs[2]:
        _tab_comparison(res)
    with tabs[3]:
        _tab_wordclouds()
    with tabs[4]:
        _tab_live()
