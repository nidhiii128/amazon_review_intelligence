"""
Baseline sentiment classification using VADER (rule-based, no training).
Evaluated against the rating-derived sentiment_label.
"""
import pathlib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]


def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(DATA_PATH)
    analyzer = SentimentIntensityAnalyzer()

    print("Scoring reviews with VADER ...")
    compounds = df["clean_text"].astype(str).apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["vader_compound"] = compounds
    df["vader_pred"] = compounds.apply(vader_label)

    acc = accuracy_score(df["sentiment_label"], df["vader_pred"])
    report = classification_report(df["sentiment_label"], df["vader_pred"], labels=LABELS, digits=3)
    cm = confusion_matrix(df["sentiment_label"], df["vader_pred"], labels=LABELS)

    out = []
    out.append("=== VADER Baseline Sentiment Classification ===")
    out.append(f"Accuracy: {acc:.4f}")
    out.append("")
    out.append("Classification report:")
    out.append(report)
    out.append("Confusion matrix (rows=true, cols=pred), label order = negative, neutral, positive:")
    out.append(str(cm))

    text = "\n".join(out)
    print(text)
    (METRICS_DIR / "vader_results.txt").write_text(text)


if __name__ == "__main__":
    main()
