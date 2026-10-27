"""Generates the schematic and CI figures used in the paper (paper/figures/).
Existing result figures (confusion matrix, curves, word clouds, ...) are
copied from reports/figures so the paper folder is self-contained."""
import json
import pathlib
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = BASE / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, ORANGE, BLUE, TEAL, GREY = "#131a2b", "#d95926", "#2563eb", "#0f766e", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, text, fc="white", ec=NAVY, fs=7.5, bold=False, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color))


def framework():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.55, 0.13, 0.34, "Amazon\nReviews 2023\nElectronics,\n50k sample", fc="#f3f4f6", fs=6.6)
    box(ax, 0.02, 0.10, 0.13, 0.34, "Product\nmetadata:\ntitle, brand,\nimage", fc="#f3f4f6", fs=6.6)
    box(ax, 0.19, 0.28, 0.12, 0.44, "Cleaning,\ndeduplication,\nrating-derived\nlabels,\nparent_asin\njoin", fc="#fff7ed", ec=ORANGE, fs=6.4)
    arrow(ax, 0.15, 0.72, 0.19, 0.56); arrow(ax, 0.15, 0.27, 0.19, 0.44)

    cols = [
        (0.355, BLUE, "NLP", ["VADER lexicon", "TF-IDF +\nLR / SVM", "LDA topics\n(k = 8)", "Aspect-based\nsentiment (ABSA)"]),
        (0.515, ORANGE, "Deep Learning", ["BiLSTM +\nAttention", "DistilBERT\nfine-tuning", "MiniLM sentence\nembeddings", "FLAN-T5\ngenerator"]),
        (0.675, TEAL, "Advanced ML", ["XGBoost\nhelpfulness", "MiniBatchKMeans\nclustering", "Isolation Forest\nsuspicious reviews", "Hyperparameter\nsearch"]),
    ]
    bus_y = 0.975
    ax.plot([0.25, 0.25], [0.72, bus_y], color=NAVY, lw=1.1)
    ax.plot([0.25, 0.675 + 0.0625], [bus_y, bus_y], color=NAVY, lw=1.1)
    for x, c, title, items in cols:
        cx = x + 0.0625
        arrow(ax, cx, bus_y, cx, 0.935, color=c)
        box(ax, x, 0.82, 0.125, 0.11, title, fc=c, ec=c, fs=6.6, bold=True, tc="white")
        for i, it in enumerate(items):
            box(ax, x, 0.62 - i * 0.185, 0.125, 0.15, it, ec=c, fs=5.9)
    box(ax, 0.865, 0.50, 0.115, 0.30, "Product\nintelligence:\naspect, trust,\nhelpfulness", fc="#f3f4f6", fs=6.4)
    box(ax, 0.865, 0.10, 0.115, 0.30, "Semantic\nsearch + RAG\nwith similarity\nfloor", fc="#f3f4f6", fs=6.4)
    arrow(ax, 0.82, 0.65, 0.855, 0.65); arrow(ax, 0.82, 0.25, 0.855, 0.25)
    arrow(ax, 0.9225, 0.495, 0.9225, 0.405)
    ax.text(0.5, 0.01, "All outputs feed one interactive dashboard (buyer view, business view, model inspection, live classification)",
            ha="center", fontsize=6.6, style="italic", color=GREY)
    fig.savefig(OUT / "fig_framework.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def lstm_arch():
    fig, ax = plt.subplots(figsize=(3.5, 3.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    layers = [
        ("Review text  (lower-cased, regex tokens)", "white", NAVY),
        ("Index lookup, vocabulary 14,937, length 200", "#f3f4f6", NAVY),
        ("Embedding layer  (128-d, learned)", "#eff6ff", BLUE),
        ("2-layer bidirectional LSTM  (128 units/dir.)", "#fff7ed", ORANGE),
        ("Attention pooling  (learned token weights)", "#fff7ed", ORANGE),
        ("Dropout 0.4 + linear layer", "#f0fdfa", TEAL),
        ("Softmax:  negative / neutral / positive", "white", NAVY),
    ]
    h, gap, y = 0.09, 0.04, 0.90
    for i, (t, fc, ec) in enumerate(layers):
        box(ax, 0.06, y, 0.88, h, t, fc=fc, ec=ec, fs=7.2)
        if i < len(layers) - 1:
            arrow(ax, 0.5, y, 0.5, y - gap + 0.003)
        y -= h + gap
    fig.savefig(OUT / "fig_lstm_arch.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def model_ci():
    res = json.loads((BASE / "reports" / "metrics" / "paper_experiments.json").read_text())["sentiment"]
    names = ["VADER", "TF-IDF + LR", "TF-IDF + SVM", "BiLSTM + Attention", "DistilBERT"]
    colors = [GREY, BLUE, BLUE, ORANGE, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=True, gridspec_kw={"wspace": 0.08})
    ypos = list(range(len(names)))[::-1]
    for ax, key, ci, title, lim in [(axes[0], "accuracy", "acc_ci", "Accuracy", (0.78, 0.94)),
                                    (axes[1], "macro_f1", "f1_ci", "Macro-F1", (0.47, 0.81))]:
        for y, n, c in zip(ypos, names, colors):
            v = res[n][key]
            ax.barh(y, v, xerr=[[v - res[n][ci][0]], [res[n][ci][1] - v]], color=c, height=0.62,
                    capsize=2, error_kw={"lw": 0.9})
            ax.text(res[n][ci][1] + 0.003, y, f"{v:.3f}", va="center", fontsize=5.8)
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(names, fontsize=6.4)
    fig.savefig(OUT / "fig_model_ci.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def dl_results():
    """Training curves + confusion matrix of the LSTM, readable at full width."""
    hist = json.loads((BASE / "reports" / "metrics" / "lstm_history.json").read_text())
    txt = (BASE / "reports" / "metrics" / "lstm_results.txt").read_text()
    nums = re.findall(r"\[\[\s*(\d+)\s+(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\]\]", txt)[0]
    cm = [[int(nums[3 * i + j]) for j in range(3)] for i in range(3)]
    ep = list(range(1, len(hist["val_f1"]) + 1))
    best = max(range(len(ep)), key=lambda i: hist["val_f1"][i]) + 1
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1, 1, 0.95]})
    axes[0].plot(ep, hist["train_loss"], marker="o", ms=3, color="#d03b3b")
    axes[0].set_title("Training loss", fontsize=8); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Weighted cross-entropy")
    axes[1].plot(ep, hist["val_acc"], marker="o", ms=3, label="Val. accuracy", color="#1f77b4")
    axes[1].plot(ep, hist["val_f1"], marker="s", ms=3, label="Val. macro-F1", color=ORANGE)
    axes[1].axvline(best, ls="--", color="gray", lw=0.8)
    axes[1].set_title(f"Validation (best epoch {best})", fontsize=8); axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=6.5, loc="center right")
    for ax in axes[:2]:
        ax.grid(alpha=0.3, lw=0.4); ax.tick_params(labelsize=7)
    labs = ["neg", "neu", "pos"]
    im = axes[2].imshow(cm, cmap="Oranges")
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(labs, fontsize=7)
    axes[2].set_yticks(range(3)); axes[2].set_yticklabels(labs, fontsize=7)
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True"); axes[2].set_title("Test confusion matrix", fontsize=8)
    mx = max(max(r) for r in cm)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=7,
                         color="white" if cm[i][j] > mx / 2 else "black")
    plt.tight_layout()
    fig.savefig(OUT / "fig_dl_results.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def aspects_and_importance():
    import joblib
    import pandas as pd
    sys.path.insert(0, str(BASE / "src" / "advanced_ml"))
    from helpfulness_xgboost import FEATURE_COLS
    a = pd.read_csv(BASE / "data" / "processed" / "absa_aspect_sentiments.csv")
    t = (pd.crosstab(a["aspect"], a["aspect_sentiment"], normalize="index") * 100)[["positive", "neutral", "negative"]]
    t = t.sort_values("negative")
    nice = {"build_quality": "Build quality", "size_comfort": "Size/comfort", "price_value": "Price/value",
            "customer_service": "Customer service"}
    labels = [nice.get(i, i.title()) for i in t.index]
    model = joblib.load(BASE / "models" / "xgboost_helpfulness.joblib")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    left = [0.0] * len(t)
    for col, c in [("positive", "#2ca02c"), ("neutral", "#ff7f0e"), ("negative", "#d62728")]:
        axes[0].barh(labels, t[col].values, left=left, color=c, label=col.title(), height=0.7)
        left = [l + v for l, v in zip(left, t[col].values)]
    axes[0].set_xlabel("% of aspect mentions", fontsize=7.5); axes[0].set_xlim(0, 100)
    axes[0].set_title("(a) Aspect-level sentiment", fontsize=8)
    axes[0].legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.45, -0.42), frameon=False)
    axes[0].tick_params(labelsize=7)
    names = {"review_length_words": "Review length", "num_aspects_mentioned": "Aspects mentioned",
             "rating": "Rating", "verified_purchase_int": "Verified purchase", "vader_compound": "VADER score",
             "review_age_days": "Review age", "title_length_words": "Title length",
             "capital_ratio": "Capital ratio", "question_count": "Question marks", "exclamation_count": "Exclamations"}
    axes[1].barh([names[i] for i in imp.index], imp.values, color=BLUE, height=0.7)
    axes[1].set_xlabel("XGBoost feature importance", fontsize=7.5)
    axes[1].set_title("(b) Helpfulness model", fontsize=8); axes[1].tick_params(labelsize=7)
    axes[1].grid(axis="x", alpha=0.3, lw=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "fig_aspects_importance.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def copy_existing():
    for name in ["lstm_confusion_matrix.png", "lstm_training_curves.png", "wordcloud_all_sentiments.png",
                 "aspect_sentiment_stacked.png", "xgboost_feature_importance.png"]:
        shutil.copy(BASE / "reports" / "figures" / name, OUT / name)


if __name__ == "__main__":
    framework(); lstm_arch(); model_ci(); dl_results(); aspects_and_importance(); copy_existing()
    print("figures written to", OUT)
