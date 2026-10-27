"""Poster-specific versions of the figures: drawn at the size they are
printed, with large lettering, so text stays legible on an A1 poster (the
paper's figures are designed for a 3.5 in column and are too small there).
Also crops dashboard screenshots to their content area.
Output: submission/assets/poster_figs/
"""
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

BASE = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = BASE / "submission" / "assets" / "poster_figs"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, ORANGE, BLUE, TEAL, GREY = "#131a2b", "#d95926", "#2563eb", "#0f766e", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc="white", ec=NAVY, fs=10, bold=False, tc=NAVY, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.8, color=color))


def framework():
    fig, ax = plt.subplots(figsize=(6.5, 4.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.84, 0.44, 0.14, "Amazon Reviews 2023\n50,000 Electronics reviews", fc="#f3f4f6", fs=10.5)
    box(ax, 0.54, 0.84, 0.44, 0.14, "Product metadata\ntitle, brand, image", fc="#f3f4f6", fs=10.5)
    box(ax, 0.12, 0.66, 0.76, 0.12, "Clean, label from star ratings, join on parent ASIN", fc="#fff7ed", ec=ORANGE, fs=10.5)
    arrow(ax, 0.24, 0.84, 0.35, 0.78); arrow(ax, 0.76, 0.84, 0.65, 0.78)
    cols = [(0.02, BLUE, "NLP", "VADER\nTF-IDF + LR / SVM\nLDA topics\nAspect sentiment"),
            (0.35, ORANGE, "Deep Learning", "BiLSTM + Attention\nDistilBERT\nMiniLM embeddings\nFLAN-T5"),
            (0.68, TEAL, "Advanced ML", "XGBoost helpfulness\nMiniBatchKMeans\nIsolation Forest\nRandomized search")]
    for x, c, t, items in cols:
        arrow(ax, 0.5, 0.66, x + 0.15, 0.585, color=c)
        box(ax, x, 0.47, 0.30, 0.10, t, fc=c, ec=c, fs=11.5, bold=True, tc="white")
        box(ax, x, 0.17, 0.30, 0.27, items, ec=c, fs=9.6)
        arrow(ax, x + 0.15, 0.17, 0.5, 0.12, color=c)
    box(ax, 0.06, 0.0, 0.88, 0.115, "Product intelligence, semantic search and grounded Q and A,\nall delivered through one interactive dashboard", fc="#f3f4f6", fs=10)
    fig.savefig(OUT / "poster_framework.png", dpi=250, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def lstm():
    fig, ax = plt.subplots(figsize=(3.3, 3.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    layers = [("Review text", "white", NAVY), ("Tokens, vocabulary 14,937", "#f3f4f6", NAVY),
              ("Embedding (128-d)", "#eff6ff", BLUE), ("2 x BiLSTM (128 units)", "#fff7ed", ORANGE),
              ("Attention pooling", "#fff7ed", ORANGE), ("Dropout 0.4 + linear", "#f0fdfa", TEAL),
              ("Softmax: neg / neu / pos", "white", NAVY)]
    h, gap, y = 0.10, 0.045, 0.90
    for i, (t, fc, ec) in enumerate(layers):
        box(ax, 0.03, y, 0.94, h, t, fc=fc, ec=ec, fs=10.5)
        if i < len(layers) - 1: arrow(ax, 0.5, y, 0.5, y - gap + 0.004)
        y -= h + gap
    fig.savefig(OUT / "poster_lstm.png", dpi=250, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def ci_chart():
    res = json.loads((BASE / "reports" / "metrics" / "paper_experiments.json").read_text())["sentiment"]
    names = ["VADER", "TF-IDF + LR", "TF-IDF + SVM", "BiLSTM + Attention", "DistilBERT"]
    colors = [GREY, BLUE, BLUE, ORANGE, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), sharey=True, gridspec_kw={"wspace": 0.08})
    ypos = list(range(5))[::-1]
    for ax, key, ci, title, lim in [(axes[0], "accuracy", "acc_ci", "Accuracy", (0.78, 0.945)), (axes[1], "macro_f1", "f1_ci", "Macro-F1", (0.47, 0.82))]:
        for y, n, c in zip(ypos, names, colors):
            v = res[n][key]
            ax.barh(y, v, xerr=[[v - res[n][ci][0]], [res[n][ci][1] - v]], color=c, height=0.62, capsize=3, error_kw={"lw": 1.2})
            ax.text(res[n][ci][1] + 0.004, y, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
        ax.set_xlim(*lim); ax.set_title(title, fontsize=12, fontweight="bold"); ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(names, fontsize=10)
    fig.savefig(OUT / "poster_ci.png", dpi=250, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def dl_results():
    hist = json.loads((BASE / "reports" / "metrics" / "lstm_history.json").read_text())
    txt = (BASE / "reports" / "metrics" / "lstm_results.txt").read_text()
    nums = re.findall(r"\[\[\s*(\d+)\s+(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\]\]", txt)[0]
    cm = [[int(nums[3 * i + j]) for j in range(3)] for i in range(3)]
    ep = list(range(1, len(hist["val_f1"]) + 1)); best = max(range(len(ep)), key=lambda i: hist["val_f1"][i]) + 1
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].plot(ep, hist["val_acc"], marker="o", ms=4, lw=2, label="Val. accuracy", color="#1f77b4")
    axes[0].plot(ep, hist["val_f1"], marker="s", ms=4, lw=2, label="Val. macro-F1", color=ORANGE)
    axes[0].axvline(best, ls="--", color="gray", lw=1.2)
    axes[0].set_title(f"Validation (best epoch {best})", fontsize=12, fontweight="bold"); axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].legend(fontsize=9, loc="center right"); axes[0].grid(alpha=0.3); axes[0].tick_params(labelsize=9)
    labs = ["neg", "neu", "pos"]
    axes[1].imshow(cm, cmap="Oranges")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(labs, fontsize=10); axes[1].set_yticks(range(3)); axes[1].set_yticklabels(labs, fontsize=10)
    axes[1].set_xlabel("Predicted", fontsize=10); axes[1].set_ylabel("True", fontsize=10)
    axes[1].set_title("Test confusion matrix", fontsize=12, fontweight="bold")
    mx = max(max(r) for r in cm)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="white" if cm[i][j] > mx / 2 else "black")
    plt.tight_layout()
    fig.savefig(OUT / "poster_dl_results.png", dpi=250, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def aspects():
    import pandas as pd
    a = pd.read_csv(BASE / "data" / "processed" / "absa_aspect_sentiments.csv")
    t = (pd.crosstab(a["aspect"], a["aspect_sentiment"], normalize="index") * 100)[["positive", "neutral", "negative"]].sort_values("negative")
    nice = {"build_quality": "Build quality", "size_comfort": "Size / comfort", "price_value": "Price / value", "customer_service": "Customer service"}
    labels = [nice.get(i, i.title()) for i in t.index]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    left = [0.0] * len(t)
    for col, c in [("positive", "#2ca02c"), ("neutral", "#ff7f0e"), ("negative", "#d62728")]:
        ax.barh(labels, t[col].values, left=left, color=c, label=col.title(), height=0.72)
        left = [l + v for l, v in zip(left, t[col].values)]
    for y, v in enumerate(t["negative"].values):
        ax.text(99, y, f"{v:.1f}%", ha="right", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_xlabel("% of aspect mentions (red share = negative)", fontsize=10); ax.tick_params(labelsize=10)
    ax.legend(fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    fig.savefig(OUT / "poster_aspects.png", dpi=250, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def crop_screens():
    src = BASE / "submission" / "assets" / "screens"
    for name in ["03b_insights_aspects", "03c_insights_trust", "07d_dl_live", "05_ask"]:
        im = Image.open(src / f"{name}.png")
        im.crop((318, 62, 1440, 900)).save(OUT / f"{name}_crop.png")


if __name__ == "__main__":
    framework(); lstm(); ci_chart(); dl_results(); aspects(); crop_screens()
    print("poster figures written to", OUT)
