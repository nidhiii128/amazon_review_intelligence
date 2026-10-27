"""Parameter counts and inference throughput of the three trained sentiment
models (SVM on CPU; BiLSTM and DistilBERT on GPU and CPU), measured on
reviews from the shared test split. Output: reports/metrics/inference_cost.json
"""
import json, pathlib, sys, time
import joblib, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

BASE = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src" / "deep_learning"))
from lstm_model import load_checkpoint, predict  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

df = pd.read_csv(BASE / "data" / "processed" / "electronics_reviews_clean.csv")
X, y = df["clean_text"].astype(str), df["sentiment_label"]
_, Xte, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
texts = Xte.tolist()[:1000]
out = {"n_reviews": len(texts)}


def timed(fn, reps=3):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return len(texts) / min(ts)


vec, svm = joblib.load(BASE / "models" / "tfidf_vectorizer.joblib"), joblib.load(BASE / "models" / "tfidf_svm.joblib")
out["svm"] = {"params": int(svm.coef_.size + svm.intercept_.size), "cpu_rps": timed(lambda: svm.predict(vec.transform(texts)))}

for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
    model, vocab, cfg = load_checkpoint(BASE / "models" / "lstm_sentiment.pt", dev)
    out.setdefault("lstm", {"params": int(sum(p.numel() for p in model.parameters()))})
    def run_lstm():
        for i in range(0, len(texts), 250):
            predict(model, vocab, cfg, texts[i:i + 250], dev)
    out["lstm"][f"{dev}_rps"] = timed(run_lstm)

tok = AutoTokenizer.from_pretrained(BASE / "models" / "bert_sentiment")
for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
    bert = AutoModelForSequenceClassification.from_pretrained(BASE / "models" / "bert_sentiment").to(dev).eval()
    out.setdefault("distilbert", {"params": int(sum(p.numel() for p in bert.parameters()))})
    n = len(texts) if dev == "cuda" else 200
    sub = texts[:n]
    def run_bert():
        with torch.no_grad():
            for i in range(0, len(sub), 64):
                enc = tok(sub[i:i + 64], return_tensors="pt", truncation=True, max_length=128, padding=True).to(dev)
                bert(**enc)
    t = []
    run_bert()
    for _ in range(2):
        if dev == "cuda": torch.cuda.synchronize()
        s = time.perf_counter(); run_bert()
        if dev == "cuda": torch.cuda.synchronize()
        t.append(time.perf_counter() - s)
    out["distilbert"][f"{dev}_rps"] = len(sub) / min(t)

(BASE / "reports" / "metrics" / "inference_cost.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
