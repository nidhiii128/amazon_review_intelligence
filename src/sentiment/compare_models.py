"""
Builds a bar chart comparing all sentiment models run so far: VADER,
TF-IDF+LR, TF-IDF+SVM, and (if it has been run) fine-tuned DistilBERT.
Safe to run at any point in the project -- models that haven't been run
yet are just skipped.
"""
import pathlib
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR = BASE_DIR / "reports" / "figures"


def macro_f1_from_report(text):
    m = re.findall(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    return float(m[0][2])


def load_vader():
    path = METRICS_DIR / "vader_results.txt"
    if not path.exists():
        return None
    text = path.read_text()
    acc = float(re.search(r"Accuracy: ([\d.]+)", text).group(1))
    f1 = macro_f1_from_report(text)
    return acc, f1


def load_tfidf():
    path = METRICS_DIR / "tfidf_classical_results.txt"
    if not path.exists():
        return None
    text = path.read_text()
    lr = re.search(r"TF-IDF \+ Logistic Regression\s+([\d.]+)\s+([\d.]+)", text)
    svm = re.search(r"TF-IDF \+ Linear SVM\s+([\d.]+)\s+([\d.]+)", text)
    results = {}
    if lr:
        results["TF-IDF +\nLogistic Regression"] = (float(lr.group(1)), float(lr.group(2)))
    if svm:
        results["TF-IDF +\nLinear SVM"] = (float(svm.group(1)), float(svm.group(2)))
    return results


def load_bert():
    path = METRICS_DIR / "bert_results.txt"
    if not path.exists():
        return None
    text = path.read_text()
    acc = float(re.search(r"Accuracy: ([\d.]+)", text).group(1))
    f1 = macro_f1_from_report(text)
    return acc, f1


def main():
    models = {}

    vader = load_vader()
    if vader:
        models["VADER\n(rule-based)"] = vader

    tfidf = load_tfidf()
    if tfidf:
        models.update(tfidf)

    bert = load_bert()
    if bert:
        models["DistilBERT\n(fine-tuned)"] = bert

    if not models:
        print("No results files found yet -- run the sentiment scripts first.")
        return

    names = list(models.keys())
    acc = [models[n][0] for n in names]
    f1 = [models[n][1] for n in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(2.4 * len(names) + 2, 5))
    plt.bar(x - width / 2, acc, width, label="Accuracy", color="#4c72b0")
    plt.bar(x + width / 2, f1, width, label="Macro F1", color="#dd8452")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Sentiment Model Comparison (Electronics reviews, 50k sample)")
    plt.legend()
    for i, v in enumerate(acc):
        plt.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(f1):
        plt.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path = FIG_DIR / "sentiment_model_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison chart ({len(names)} models) to {out_path}")


if __name__ == "__main__":
    main()
