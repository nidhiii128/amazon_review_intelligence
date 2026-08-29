"""
Aspect-Based Sentiment Analysis (ABSA).

For each review, finds sentences that mention a known product aspect
(battery, camera, display, sound, performance, build quality, price,
connectivity, size/comfort, customer service) and scores that sentence's
sentiment with VADER. This gives per-aspect sentiment instead of one
overall rating, e.g. "camera positive, battery negative" from one review.

Sentence splitting is a lightweight regex (no NLTK download dependency,
so it runs the same way on any machine without extra setup).

Outputs:
  - data/processed/absa_aspect_sentiments.csv   long-format: one row per
        (review_id, aspect, aspect_sentiment) mention
  - reports/metrics/absa_summary.txt            aspect x sentiment % table
  - reports/figures/aspect_sentiment_stacked.png stacked bar chart
"""
import re
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "absa_aspect_sentiments.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ASPECT_KEYWORDS = {
    "battery": ["battery", "batteries", "charge", "charging", "charger", "power drain", "drains", "runtime"],
    "camera": ["camera", "photo", "photos", "picture", "pictures", "lens", "video quality", "image quality"],
    "display": ["screen", "display", "resolution", "brightness", "touchscreen", "monitor"],
    "sound": ["sound", "audio", "speaker", "speakers", "volume", "bass", "microphone", "mic"],
    "performance": ["performance", "speed", "lag", "laggy", "fast", "slow", "processor", "responsive", "freeze", "freezing"],
    "build_quality": ["build quality", "durable", "durability", "sturdy", "flimsy", "cheaply made", "well made", "material"],
    "price_value": ["price", "value", "expensive", "cheap", "overpriced", "worth it", "worth the", "cost"],
    "connectivity": ["bluetooth", "wifi", "wi-fi", "connectivity", "connection", "pairing", "connects", "signal"],
    "size_comfort": ["size", "fit", "comfortable", "comfort", "weight", "heavy", "lightweight", "bulky", "compact"],
    "customer_service": ["customer service", "support", "warranty", "return policy", "refund", "replacement"],
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(DATA_PATH)
    analyzer = SentimentIntensityAnalyzer()

    rows = []
    print(f"Scanning {len(df):,} reviews for aspect mentions ...")
    for _, row in df.iterrows():
        sentences = split_sentences(row["clean_text"])
        text_lower_sentences = [(s, s.lower()) for s in sentences]
        for aspect, keywords in ASPECT_KEYWORDS.items():
            matched_sentences = [
                s for s, s_lower in text_lower_sentences
                if any(kw in s_lower for kw in keywords)
            ]
            if not matched_sentences:
                continue
            combined = " ".join(matched_sentences)
            compound = analyzer.polarity_scores(combined)["compound"]
            rows.append({
                "review_id": row["review_id"],
                "asin": row["asin"],
                "aspect": aspect,
                "aspect_sentiment": sentiment_label(compound),
                "aspect_compound": compound,
                "evidence": combined[:300],
            })

    absa_df = pd.DataFrame(rows)
    absa_df.to_csv(OUT_PATH, index=False)
    print(f"Found {len(absa_df):,} aspect mentions across {absa_df['review_id'].nunique():,} reviews")
    print(f"Saved to {OUT_PATH}")

    # Aspect x sentiment percentage table
    pct_table = (
        pd.crosstab(absa_df["aspect"], absa_df["aspect_sentiment"], normalize="index") * 100
    ).round(1)
    for col in ["positive", "neutral", "negative"]:
        if col not in pct_table.columns:
            pct_table[col] = 0.0
    pct_table = pct_table[["positive", "neutral", "negative"]]
    counts = absa_df["aspect"].value_counts()
    pct_table["mention_count"] = counts

    pct_table = pct_table.sort_values("negative", ascending=False)

    summary_lines = [
        "=== Aspect-Based Sentiment Summary ===",
        "(% of mentions of that aspect that are positive / neutral / negative)",
        "",
        pct_table.to_string(),
        "",
        "Aspects sorted by highest negative-sentiment share first -- these are",
        "the recurring complaint areas a company should prioritize.",
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (METRICS_DIR / "absa_summary.txt").write_text(summary_text)

    # Stacked bar chart
    plot_df = pct_table[["positive", "neutral", "negative"]].sort_values("negative")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"positive": "#2ca02c", "neutral": "#ff7f0e", "negative": "#d62728"}
    left = pd.Series(0.0, index=plot_df.index)
    for col in ["positive", "neutral", "negative"]:
        ax.barh(plot_df.index, plot_df[col], left=left, label=col, color=colors[col])
        left += plot_df[col]
    ax.set_xlabel("% of mentions")
    ax.set_title("Aspect-Level Sentiment (Electronics reviews)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "aspect_sentiment_stacked.png", dpi=150)
    plt.close()
    print(f"Saved chart to {FIG_DIR / 'aspect_sentiment_stacked.png'}")


if __name__ == "__main__":
    main()
