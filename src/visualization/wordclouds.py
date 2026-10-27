"""
Word cloud visualisations of review language, one per sentiment class.

Plain word frequency makes every cloud look the same ("great", "good",
"use", "product"), so words are weighted by frequency within the class
multiplied by how over-represented they are in that class relative to the
whole corpus. That surfaces the vocabulary that actually characterises
negative / neutral / positive reviews ("refund", "stopped", "returned"
vs "love", "perfect", "excellent").

Outputs (reports/figures/):
  wordcloud_negative.png, wordcloud_neutral.png, wordcloud_positive.png,
  wordcloud_all_sentiments.png (the three side by side)
"""
import pathlib
from collections import Counter
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_STOPWORDS = {"just", "really", "like", "get", "got", "one", "use", "used", "using", "t", "s", "m", "ve",
                   "don", "didn", "doesn", "isn", "wasn", "stars", "star", "amazon", "item", "product"}
STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS
TOKEN_RE = re.compile(r"[a-z]{3,}")

# Saturated, dark-enough shades only: a continuous colormap gives the most
# frequent (largest) words the palest colour, which washes them out against
# the white background.
CLASS_STYLE = {
    "negative": {"palette": ["#7f1d1d", "#991b1b", "#b91c1c", "#c2410c", "#9a3412"], "bg": "white"},
    "neutral": {"palette": ["#78350f", "#92400e", "#b45309", "#a16207", "#854d0e"], "bg": "white"},
    "positive": {"palette": ["#14532d", "#166534", "#15803d", "#0f766e", "#047857"], "bg": "white"},
}


def make_color_func(palette):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return palette[sum(map(ord, word)) % len(palette)]
    return color_func


def class_counts(texts):
    c = Counter()
    for t in texts:
        c.update(w for w in TOKEN_RE.findall(str(t).lower()) if w not in STOPWORDS)
    return c


def distinctive_weights(class_counter, overall_counter, min_count=15):
    total_class = sum(class_counter.values())
    total_all = sum(overall_counter.values())
    weights = {}
    for word, n in class_counter.items():
        if n < min_count:
            continue
        class_freq = n / total_class
        lift = class_freq / (overall_counter[word] / total_all)
        weights[word] = class_freq * (lift ** 0.7)
    return weights


def main():
    df = pd.read_csv(DATA_PATH)
    overall = class_counts(df["clean_text"])
    clouds = {}
    for label, style in CLASS_STYLE.items():
        texts = df.loc[df["sentiment_label"] == label, "clean_text"]
        weights = distinctive_weights(class_counts(texts), overall)
        wc = WordCloud(
            width=900, height=520, background_color=style["bg"],
            color_func=make_color_func(style["palette"]),
            max_words=90, prefer_horizontal=0.9, random_state=42,
        ).generate_from_frequencies(weights)
        clouds[label] = (wc, len(texts))
        wc.to_file(str(FIG_DIR / f"wordcloud_{label}.png"))
        print(f"{label}: {len(texts):,} reviews, top words: {list(sorted(weights, key=weights.get, reverse=True)[:8])}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for ax, (label, (wc, n)) in zip(axes, clouds.items()):
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label.title()} reviews (n={n:,})", fontsize=14, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wordcloud_all_sentiments.png", dpi=130)
    plt.close()
    print(f"Saved word clouds to {FIG_DIR}")


if __name__ == "__main__":
    main()
