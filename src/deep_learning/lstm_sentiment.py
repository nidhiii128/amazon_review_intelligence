"""
Deep learning sentiment classifier: bidirectional LSTM with attention
(PyTorch), trained from scratch on the Electronics review sample to
classify reviews as positive / neutral / negative.

Evaluated on the exact same stratified 80/20 test split (random_state=42)
used by the TF-IDF models and DistilBERT, so the comparison is like for
like. 10% of the training portion is held out as a validation set for
early stopping (best epoch chosen by validation macro-F1, since the
classes are heavily imbalanced and accuracy alone is misleading). The loss
uses square-root-inverse-frequency class weights for the same reason.

Outputs:
  - models/lstm_sentiment.pt                     weights + vocab + config
  - reports/metrics/lstm_results.txt             accuracy, macro-F1, report
  - reports/figures/lstm_training_curves.png     loss / val accuracy / val F1
  - reports/figures/lstm_confusion_matrix.png
  - data/processed/lstm_predictions.csv          prediction for every review
        (with a train/val/test flag so in-sample predictions are never
        mistaken for held-out ones)
"""
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lstm_model import BiLSTMAttention, DEFAULT_CONFIG, LABELS, build_vocab, encode  # noqa: E402

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
PRED_PATH = BASE_DIR / "data" / "processed" / "lstm_predictions.csv"
MODEL_PATH = BASE_DIR / "models" / "lstm_sentiment.pt"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR = BASE_DIR / "reports" / "figures"
for d in (METRICS_DIR, FIG_DIR, MODEL_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 12
PATIENCE = 3
LR = 2e-3

torch.manual_seed(SEED)
np.random.seed(SEED)


def to_tensors(texts, vocab, max_len):
    ids, lens = zip(*(encode(t, vocab, max_len) for t in texts))
    return torch.tensor(ids), torch.tensor(lens)


@torch.no_grad()
def predict_all(model, x, lengths, device):
    model.eval()
    probs = []
    for i in range(0, len(x), 1024):
        logits = model(x[i:i + 1024].to(device), lengths[i:i + 1024])
        probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probs).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = dict(DEFAULT_CONFIG)

    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"].astype(str)
    y = df["sentiment_label"]

    # identical call to tfidf_classical.py / bert_finetune.py -> identical test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    print(f"train {len(X_tr):,} | val {len(X_val):,} | test {len(X_test):,}")

    vocab = build_vocab(X_tr, cfg["max_vocab"], cfg["min_freq"])
    print(f"Vocabulary size: {len(vocab):,}")
    label2id = {l: i for i, l in enumerate(LABELS)}

    xtr, ltr = to_tensors(X_tr, vocab, cfg["max_len"])
    xva, lva = to_tensors(X_val, vocab, cfg["max_len"])
    xte, lte = to_tensors(X_test, vocab, cfg["max_len"])
    ytr = torch.tensor(y_tr.map(label2id).values)
    yva = y_val.map(label2id).values
    yte = y_test.map(label2id).values

    model = BiLSTMAttention(
        len(vocab), cfg["embed_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")

    counts = np.bincount(ytr.numpy(), minlength=3)
    weights = torch.tensor((len(ytr) / (3 * counts)) ** 0.5, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(xtr, ltr, ytr), batch_size=BATCH_SIZE, shuffle=True)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, bad_epochs = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, lb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        val_pred = predict_all(model, xva, lva, device).argmax(1)
        acc = accuracy_score(yva, val_pred)
        f1 = f1_score(yva, val_pred, average="macro")
        history["train_loss"].append(total / len(ytr))
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        print(f"epoch {epoch:2d} | loss {history['train_loss'][-1]:.4f} | val acc {acc:.4f} | val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, bad_epochs = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping (no val macro-F1 gain for {PATIENCE} epochs)")
                break
    train_minutes = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    best_epoch = int(np.argmax(history["val_f1"])) + 1

    test_probs = predict_all(model, xte, lte, device)
    test_pred = test_probs.argmax(1)
    acc = accuracy_score(yte, test_pred)
    f1_macro = f1_score(yte, test_pred, average="macro")
    report = classification_report(yte, test_pred, target_names=LABELS, digits=3)
    cm = confusion_matrix(yte, test_pred)

    result = (
        "=== BiLSTM + Attention Sentiment Classification (PyTorch) ===\n"
        f"Device: {device}\n"
        f"Train size: {len(X_tr):,} | Validation size: {len(X_val):,} | Test size: {len(X_test):,}\n"
        f"Vocabulary: {len(vocab):,} tokens | Max length: {cfg['max_len']} | "
        f"Embedding: {cfg['embed_dim']} | Hidden: {cfg['hidden_dim']} x {cfg['num_layers']} layers (bidirectional)\n"
        f"Trainable parameters: {n_params:,}\n"
        f"Epochs run: {len(history['val_f1'])} (best epoch by val macro-F1: {best_epoch}) | "
        f"Training time: {train_minutes:.1f} minutes\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro F1: {f1_macro:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=true, cols=pred), label order = {', '.join(LABELS)}:\n{cm}\n"
    )
    print(result)
    (METRICS_DIR / "lstm_results.txt").write_text(result)

    torch.save({"state_dict": best_state, "vocab": vocab, "config": cfg, "history": history}, MODEL_PATH)
    (METRICS_DIR / "lstm_history.json").write_text(json.dumps(history))

    # --- training curves ---
    epochs = range(1, len(history["val_f1"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", color="#d03b3b")
    axes[0].set_title("Training loss (class-weighted cross-entropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Validation accuracy", color="#1f77b4")
    axes[1].plot(epochs, history["val_f1"], marker="s", label="Validation macro-F1", color="#eb6834")
    axes[1].axvline(best_epoch, ls="--", color="gray", label=f"Best epoch ({best_epoch})")
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lstm_training_curves.png", dpi=150)
    plt.close()

    # --- confusion matrix ---
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("BiLSTM + Attention -- test set")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lstm_confusion_matrix.png", dpi=150)
    plt.close()

    # --- predictions for every review, flagged by split ---
    all_x, all_l = to_tensors(X, vocab, cfg["max_len"])
    probs = predict_all(model, all_x, all_l, device)
    split = pd.Series("train", index=df.index)
    split.loc[X_val.index] = "val"
    split.loc[X_test.index] = "test"
    out = pd.DataFrame({
        "review_id": df["review_id"],
        "lstm_label": [LABELS[i] for i in probs.argmax(1)],
        "lstm_confidence": probs.max(1).round(4),
        "split": split.values,
    })
    out.to_csv(PRED_PATH, index=False)
    print(f"Saved model, curves, confusion matrix and {len(out):,} predictions.")


if __name__ == "__main__":
    main()
