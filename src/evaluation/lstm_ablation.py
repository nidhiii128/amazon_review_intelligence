"""
Ablation for the paper: is the BiLSTM's macro-F1 advantage due to attention
pooling, to class-weighted loss, or neither?  Three variants x three seeds,
all on the identical train/val/test split (only the torch seed changes):

  full          attention pooling + sqrt class weights   (the reported model)
  no_weights    attention pooling, unweighted loss
  mean_pool     mean pooling,      sqrt class weights

Output: reports/metrics/lstm_ablation.json
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "deep_learning"))
from lstm_model import BiLSTMAttention, DEFAULT_CONFIG, LABELS, build_vocab, encode  # noqa: E402

OUT = BASE_DIR / "reports" / "metrics" / "lstm_ablation.json"
VARIANTS = {
    "full": {"pooling": "attention", "weights": True},
    "no_weights": {"pooling": "attention", "weights": False},
    "mean_pool": {"pooling": "mean", "weights": True},
}
SEEDS = [42, 43, 44]


def tens(texts, vocab, L):
    ids, lens = zip(*(encode(t, vocab, L) for t in texts))
    return torch.tensor(ids), torch.tensor(lens)


@torch.no_grad()
def pred(model, x, l, dev):
    model.eval()
    out = []
    for i in range(0, len(x), 1024):
        out.append(model(x[i:i + 1024].to(dev), l[i:i + 1024]).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(DEFAULT_CONFIG)
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv")
    X, y = df["clean_text"].astype(str), df["sentiment_label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.1, random_state=42, stratify=ytr)
    vocab = build_vocab(Xt, cfg["max_vocab"], cfg["min_freq"])
    l2i = {l: i for i, l in enumerate(LABELS)}
    xt, lt = tens(Xt, vocab, cfg["max_len"]); xv, lv = tens(Xv, vocab, cfg["max_len"]); xe, le = tens(Xte, vocab, cfg["max_len"])
    yt_t = torch.tensor(yt.map(l2i).values); yv_n = yv.map(l2i).values; ye_n = yte.map(l2i).values
    counts = np.bincount(yt_t.numpy(), minlength=3)
    cw = torch.tensor((len(yt_t) / (3 * counts)) ** 0.5, dtype=torch.float32, device=dev)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, v in VARIANTS.items():
        for seed in SEEDS:
            key = f"{name}_seed{seed}"
            if key in results:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            model = BiLSTMAttention(len(vocab), cfg["embed_dim"], cfg["hidden_dim"], cfg["num_layers"],
                                    cfg["dropout"], pooling=v["pooling"]).to(dev)
            crit = nn.CrossEntropyLoss(weight=cw if v["weights"] else None)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
            loader = DataLoader(TensorDataset(xt, lt, yt_t), batch_size=128, shuffle=True)
            best, state, bad = -1.0, None, 0
            t0 = time.time()
            for ep in range(12):
                model.train()
                for xb, lb, yb in loader:
                    opt.zero_grad()
                    crit(model(xb.to(dev), lb), yb.to(dev)).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                f1 = f1_score(yv_n, pred(model, xv, lv, dev), average="macro")
                if f1 > best:
                    best, bad = f1, 0
                    state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 3:
                        break
            model.load_state_dict(state)
            p = pred(model, xe, le, dev)
            results[key] = {
                "variant": name, "seed": seed, "accuracy": float(accuracy_score(ye_n, p)),
                "macro_f1": float(f1_score(ye_n, p, average="macro")),
                "f1_neutral": float(f1_score(ye_n, p, average=None)[1]), "val_f1": float(best),
                "minutes": (time.time() - t0) / 60,
            }
            OUT.write_text(json.dumps(results, indent=2))
            print(key, results[key], flush=True)


if __name__ == "__main__":
    main()
