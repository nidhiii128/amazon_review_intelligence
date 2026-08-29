"""
Exploratory Data Analysis for the processed Electronics review sample.
Produces summary stats (reports/metrics/eda_summary.txt) and figures
(reports/figures/*.png).
"""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    lines = []
    lines.append(f"Total reviews: {len(df):,}")
    lines.append(f"Unique products (asin): {df['asin'].nunique():,}")
    lines.append(f"Unique users: {df['user_id'].nunique():,}")
    lines.append(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    lines.append(f"Verified purchase rate: {df['verified_purchase'].mean():.2%}")
    lines.append(f"Average rating: {df['rating'].mean():.2f}")
    lines.append(f"Average review length (words): {df['review_length_words'].mean():.1f}")
    lines.append("")
    lines.append("Rating distribution:")
    lines.append(df["rating"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Sentiment label distribution (derived from rating):")
    lines.append(df["sentiment_label"].value_counts().to_string())
    lines.append("")
    lines.append("Helpful vote stats:")
    lines.append(df["helpful_vote"].describe().to_string())

    summary_text = "\n".join(lines)
    (METRICS_DIR / "eda_summary.txt").write_text(summary_text)
    print(summary_text)

    # 1. Rating distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="rating", data=df, hue="rating", palette="viridis", legend=False)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rating_distribution.png", dpi=150)
    plt.close()

    # 2. Sentiment label distribution
    plt.figure(figsize=(6, 4))
    order = ["positive", "neutral", "negative"]
    sns.countplot(x="sentiment_label", data=df, order=order, hue="sentiment_label",
                  palette={"positive": "#2ca02c", "neutral": "#ff7f0e", "negative": "#d62728"},
                  legend=False)
    plt.title("Derived Sentiment Label Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sentiment_distribution.png", dpi=150)
    plt.close()

    # 3. Review length distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["review_length_words"].clip(upper=300), bins=50, kde=True, color="#4c72b0")
    plt.title("Review Length Distribution (words, clipped at 300)")
    plt.xlabel("Words per review")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "review_length_distribution.png", dpi=150)
    plt.close()

    # 4. Reviews over time
    plt.figure(figsize=(8, 4))
    monthly = df.set_index("timestamp").resample("ME").size()
    monthly.plot()
    plt.title("Number of Reviews Over Time")
    plt.xlabel("Date")
    plt.ylabel("Review count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reviews_over_time.png", dpi=150)
    plt.close()

    # 5. Helpful votes vs rating
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="rating", y="helpful_vote", data=df, hue="rating", palette="coolwarm", legend=False,
                showfliers=False)
    plt.title("Helpful Votes by Rating (outliers hidden)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "helpful_votes_by_rating.png", dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Summary saved to {METRICS_DIR / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()
