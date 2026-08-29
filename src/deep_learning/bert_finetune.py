"""
Milestone 3: fine-tune a Transformer (DistilBERT) for review sentiment
classification, and compare it against the Milestone 1 baselines
(VADER, TF-IDF+LR, TF-IDF+SVM) on the exact same test split.

Why DistilBERT instead of full BERT: it keeps ~97% of BERT's language
understanding at ~60% of the size and roughly 2x the speed, which matters
a lot when this may run on a CPU-only laptop. This is a deliberate
engineering trade-off, not a shortcut -- worth stating explicitly if asked
"why not BERT itself" (swap MODEL_NAME to "bert-base-uncased" for the full
version if you have a GPU and want the heavier model).

This script needs `torch` + `transformers` (see requirements-deep-learning.txt)
and will download the pretrained DistilBERT weights from Hugging Face the
first time it runs (a few hundred MB) -- that step needs a normal internet
connection.

Automatically uses a GPU if one is available (much faster, larger sample);
falls back to a smaller training sample on CPU-only machines so it still
finishes in a reasonable time.
"""
import time
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
LABELS = ["negative", "neutral", "positive"]
MAX_LENGTH = 128


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def tokenize(tokenizer, texts):
    return tokenizer(
        list(texts), truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"].astype(str)
    y_str = df["sentiment_label"]

    # Identical split call to src/sentiment/tfidf_classical.py so the test
    # set (and therefore the comparison) is exactly the same rows.
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X, y_str, test_size=0.2, random_state=42, stratify=y_str
    )

    if device.type == "cuda":
        epochs, batch_size = 3, 16
        train_sample_n = len(X_train)
    else:
        epochs, batch_size = 2, 8
        train_sample_n = min(6000, len(X_train))
        print(f"No GPU detected -- training on a {train_sample_n:,}-review "
              f"subsample ({epochs} epochs) to keep CPU runtime reasonable. "
              f"Full test set ({len(X_test):,} reviews) is still used for evaluation.")

    if train_sample_n < len(X_train):
        X_train_sub, _, y_train_str_sub, _ = train_test_split(
            X_train, y_train_str, train_size=train_sample_n, random_state=42, stratify=y_train_str
        )
    else:
        X_train_sub, y_train_str_sub = X_train, y_train_str

    y_train = y_train_str_sub.map(LABEL_MAP).to_numpy()
    y_test = y_test_str.map(LABEL_MAP).to_numpy()

    print(f"Loading tokenizer + model: {MODEL_NAME} (downloads on first run) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    print("Tokenizing ...")
    train_encodings = tokenize(tokenizer, X_train_sub)
    test_encodings = tokenize(tokenizer, X_test)

    train_dataset = ReviewDataset(train_encodings, y_train)
    test_dataset = ReviewDataset(test_encodings, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print(f"Training for {epochs} epoch(s) on {len(train_dataset):,} reviews "
          f"(batch size {batch_size}) ...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch + 1}/{epochs}, batch {i + 1}/{len(train_loader)}, "
                      f"avg loss so far: {total_loss / (i + 1):.4f}")
        print(f"Epoch {epoch + 1}/{epochs} done. Average loss: {total_loss / len(train_loader):.4f}")

    print("Evaluating on held-out test set ...")
    y_true, y_pred = evaluate(model, test_loader, device)
    y_true_labels = [ID_TO_LABEL[i] for i in y_true]
    y_pred_labels = [ID_TO_LABEL[i] for i in y_pred]

    acc = accuracy_score(y_true_labels, y_pred_labels)
    f1_macro = f1_score(y_true_labels, y_pred_labels, labels=LABELS, average="macro")
    report = classification_report(y_true_labels, y_pred_labels, labels=LABELS, digits=3)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=LABELS)

    elapsed = time.time() - t_start
    result_text = (
        f"=== DistilBERT Fine-tuned Sentiment Classification ===\n"
        f"Device: {device}\n"
        f"Train size: {len(train_dataset):,} | Test size: {len(test_dataset):,}\n"
        f"Epochs: {epochs} | Batch size: {batch_size} | Max seq length: {MAX_LENGTH}\n"
        f"Total time: {elapsed / 60:.1f} minutes\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro F1: {f1_macro:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=true, cols=pred), label order = negative, neutral, positive:\n{cm}\n"
    )
    print(result_text)
    (METRICS_DIR / "bert_results.txt").write_text(result_text)

    save_dir = MODELS_DIR / "bert_sentiment"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")
    print(f"Results saved to {METRICS_DIR / 'bert_results.txt'}")


if __name__ == "__main__":
    main()
