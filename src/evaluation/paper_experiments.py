"""
Experiments reported in the paper, run from the saved models so every
number is reproducible:

 1. All five sentiment models re-scored on the SAME 10,000-review test split
    (VADER's original report scored all 50k reviews), with 95% bootstrap
    confidence intervals and paired bootstrap differences.
 2. Retrieval quality: semantic (MiniLM) vs TF-IDF retrieval on paraphrased
    aspect queries that deliberately avoid the aspect lexicon.
 3. The RAG similarity-floor guardrail: top-1 similarity for answerable
    vs unanswerable product-scoped questions.

Outputs: reports/metrics/paper_experiments.json and .txt
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "deep_learning"))
sys.path.insert(0, str(BASE_DIR / "src" / "absa"))
from lstm_model import load_checkpoint, predict  # noqa: E402
from aspect_sentiment import ASPECT_KEYWORDS  # noqa: E402

DATA = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
MODELS = BASE_DIR / "models"
OUT = BASE_DIR / "reports" / "metrics"
LABELS = ["negative", "neutral", "positive"]
RNG = np.random.default_rng(42)
N_BOOT = 2000


def macro_f1_fast(y, p):
    f = []
    for c in range(3):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def sentiment_experiment():
    df = pd.read_csv(DATA)
    X, y = df["clean_text"].astype(str), df["sentiment_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    texts = X_test.tolist()
    yt = y_test.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    preds = {}

    an = SentimentIntensityAnalyzer()
    comp = np.array([an.polarity_scores(t)["compound"] for t in texts])
    preds["VADER"] = np.where(comp >= 0.05, 2, np.where(comp <= -0.05, 0, 1))

    vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    Xv = vec.transform(texts)
    for name, f in [("TF-IDF + LR", "tfidf_logreg.joblib"), ("TF-IDF + SVM", "tfidf_svm.joblib")]:
        p = joblib.load(MODELS / f).predict(Xv)
        preds[name] = np.array([LABELS.index(s) for s in p])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, cfg = load_checkpoint(MODELS / "lstm_sentiment.pt", device)
    lp = []
    for i in range(0, len(texts), 1000):
        probs, _ = predict(model, vocab, cfg, texts[i:i + 1000], device)
        lp.append(probs.argmax(1))
    preds["BiLSTM + Attention"] = np.concatenate(lp)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS / "bert_sentiment")
    bert = AutoModelForSequenceClassification.from_pretrained(MODELS / "bert_sentiment").to(device).eval()
    bp = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i + 128], return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            bp.append(bert(**enc).logits.argmax(1).cpu().numpy())
    preds["DistilBERT"] = np.concatenate(bp)

    n = len(yt)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = {k: np.array([macro_f1_fast(yt[i], p[i]) for i in idx]) for k, p in preds.items()}
    boot_acc = {k: np.array([np.mean(yt[i] == p[i]) for i in idx]) for k, p in preds.items()}

    res = {}
    for k, p in preds.items():
        res[k] = {
            "accuracy": float(accuracy_score(yt, p)), "macro_f1": float(f1_score(yt, p, average="macro")),
            "acc_ci": [float(np.percentile(boot_acc[k], 2.5)), float(np.percentile(boot_acc[k], 97.5))],
            "f1_ci": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))],
            "per_class_f1": [float(x) for x in f1_score(yt, p, average=None)],
        }
    pairs = {}
    for a, b in [("BiLSTM + Attention", "TF-IDF + SVM"), ("BiLSTM + Attention", "DistilBERT"),
                 ("DistilBERT", "TF-IDF + SVM")]:
        d = boot[a] - boot[b]
        pairs[f"{a} - {b}"] = {
            "mean_diff": float(d.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_a_better": float((d > 0).mean()),
        }
        da = boot_acc[a] - boot_acc[b]
        pairs[f"{a} - {b} (accuracy)"] = {
            "mean_diff": float(da.mean()), "ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "p_a_better": float((da > 0).mean()),
        }
    return res, pairs, len(yt)


PARAPHRASE_QUERIES = {
    "battery": "I have to plug it in twice a day or it just dies",
    "camera": "images look grainy and fuzzy in dim light",
    "display": "text is hard to read and the colors look washed out",
    "sound": "voices are muffled and tinny with no low end",
    "performance": "it hangs and stutters and takes forever to respond",
    "build_quality": "feels like brittle plastic and cracked after one drop",
    "price_value": "I overpaid for what you actually get",
    "connectivity": "keeps dropping off my home network and will not link up",
    "size_comfort": "huge and awkward to hold and hurts after an hour",
    "customer_service": "the seller never answered my emails after it arrived broken",
}


def retrieval_experiment():
    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(DATA)
    emb = np.load(MODELS / "embeddings" / "review_embeddings.npy")
    ids = np.load(MODELS / "embeddings" / "review_ids.npy")
    df = df.set_index("review_id").loc[ids].reset_index()
    texts = df["clean_text"].astype(str)
    lower = texts.str.lower()

    st = SentenceTransformer("all-MiniLM-L6-v2")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    Xt = tfidf.fit_transform(texts)

    out = {}
    for aspect, q in PARAPHRASE_QUERIES.items():
        kws = ASPECT_KEYWORDS[aspect]
        # sanity: the query must not contain any aspect keyword
        assert not any(k in q.lower() for k in kws), (aspect, q)
        qv = st.encode([q], normalize_embeddings=True)[0]
        top_e = np.argsort(emb @ qv)[::-1][:5]
        qt = tfidf.transform([q])
        top_t = np.argsort((Xt @ qt.T).toarray().ravel())[::-1][:5]
        rel = lambda top: float(np.mean([any(k in lower.iloc[i] for k in kws) for i in top]))
        out[aspect] = {"semantic": rel(top_e), "tfidf": rel(top_t)}
    sem = float(np.mean([v["semantic"] for v in out.values()]))
    lex = float(np.mean([v["tfidf"] for v in out.values()]))
    return out, sem, lex


def guardrail_experiment():
    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(DATA)
    emb = np.load(MODELS / "embeddings" / "review_embeddings.npy")
    ids = np.load(MODELS / "embeddings" / "review_ids.npy")
    df = df.set_index("review_id").loc[ids].reset_index()
    st = SentenceTransformer("all-MiniLM-L6-v2")
    top_products = df["parent_asin"].value_counts()
    prods = top_products[top_products >= 60].index[:40]

    answerable = [
        "What do customers say about the battery life?", "Is the sound quality good?",
        "How is the build quality and durability?", "Is it good value for the price?",
        "Do people have problems with connectivity or pairing?",
    ]
    off_topic = [
        "What is the capital of France?", "Give me a recipe for chocolate cake.",
        "Who won the football world cup in 2010?", "How do I train a puppy to sit?",
        "What is the weather like in Paris today?",
    ]
    plausible = [
        "Does it come with a three year warranty?", "Is it available in pink?",
        "Can it be shipped to Antarctica?", "Is it certified waterproof to 50 meters?",
        "Does it support wireless charging?",
    ]
    threshold = 0.30
    res = {}
    for name, qs in [("answerable", answerable), ("plausible_unanswerable", plausible), ("off_topic", off_topic)]:
        qv = st.encode(qs, normalize_embeddings=True)
        tops = []
        for p in prods:
            mask = (df["parent_asin"] == p).to_numpy()
            sims = emb[mask] @ qv.T
            tops.extend(sims.max(axis=0).tolist())
        tops = np.array(tops)
        res[name] = {"mean_top1": float(tops.mean()), "median_top1": float(np.median(tops)),
                     "frac_below_threshold": float((tops < threshold).mean()), "n": int(len(tops))}
    return res, int(len(prods)), threshold


def main():
    results = {}
    print("1/3 sentiment models on the shared test split ...")
    results["sentiment"], results["paired"], results["n_test"] = sentiment_experiment()
    print("2/3 retrieval experiment ...")
    per, sem, lex = retrieval_experiment()
    results["retrieval"] = {"per_aspect": per, "semantic_mean": sem, "tfidf_mean": lex}
    print("3/3 guardrail experiment ...")
    g, n_prod, thr = guardrail_experiment()
    results["guardrail"] = {"groups": g, "n_products": n_prod, "threshold": thr}

    (OUT / "paper_experiments.json").write_text(json.dumps(results, indent=2))
    lines = [f"Shared test split: n={results['n_test']}", ""]
    for k, v in results["sentiment"].items():
        lines.append(f"{k:22s} acc {v['accuracy']:.4f} [{v['acc_ci'][0]:.4f},{v['acc_ci'][1]:.4f}]  "
                     f"macroF1 {v['macro_f1']:.4f} [{v['f1_ci'][0]:.4f},{v['f1_ci'][1]:.4f}]  "
                     f"perclass F1 {[round(x, 3) for x in v['per_class_f1']]}")
    lines.append("")
    for k, v in results["paired"].items():
        lines.append(f"{k}: diff {v['mean_diff']:+.4f} CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] P(a>b)={v['p_a_better']:.3f}")
    lines += ["", f"Retrieval aspect-mention@5 (paraphrase queries): semantic {sem:.3f} vs tfidf {lex:.3f}"]
    for a, v in per.items():
        lines.append(f"  {a:18s} semantic {v['semantic']:.1f}  tfidf {v['tfidf']:.1f}")
    lines += ["", f"Guardrail (top-1 cosine across {n_prod} products, threshold {thr}):"]
    for k, v in g.items():
        lines.append(f"  {k:24s} mean {v['mean_top1']:.3f} median {v['median_top1']:.3f} below-threshold {v['frac_below_threshold']:.1%} (n={v['n']})")
    (OUT / "paper_experiments.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
