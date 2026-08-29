"""
Milestone 4 (primary task): predict whether a review will be found
"helpful" (helpful_vote >= 1) vs "less helpful" (0 helpful votes), using
engineered features + XGBoost with hyperparameter tuning.

This demonstrates feature engineering, gradient boosting / ensemble
learning, and hyperparameter search -- the "advanced ML" trio called for
in the project brief -- without piling on extra unrelated algorithms.

Label: helpful_vote is heavily right-skewed (median 0, 75th pct 1), so a
binary >=1 threshold is used rather than a raw regression target; this
keeps the positive class large enough (~25-30% of reviews) to model well
with class weighting.

Needs `xgboost` (pip install xgboost). Everything else it needs
(pandas, numpy, sklearn, matplotlib, vaderSentiment) is already installed
from earlier milestones.
"""
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score,
)
from xgboost import XGBClassifier

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
ABSA_PATH = BASE_DIR / "data" / "processed" / "absa_aspect_sentiments.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
FIG_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
for d in (METRICS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rating", "review_length_words", "title_length_words", "verified_purchase_int",
    "exclamation_count", "question_count", "capital_ratio", "vader_compound",
    "review_age_days", "num_aspects_mentioned",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    max_ts = df["timestamp"].max()
    df["review_age_days"] = (max_ts - df["timestamp"]).dt.days.fillna(0)

    df["title_length_words"] = df["title"].fillna("").astype(str).str.split().str.len()
    df["verified_purchase_int"] = df["verified_purchase"].astype(int)
    df["exclamation_count"] = df["text"].fillna("").astype(str).str.count(r"!")
    df["question_count"] = df["text"].fillna("").astype(str).str.count(r"\?")

    def capital_ratio(text):
        text = str(text)
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    df["capital_ratio"] = df["text"].fillna("").apply(capital_ratio)

    print("Scoring VADER compound sentiment (used as a feature, not the target) ...")
    analyzer = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["clean_text"].astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    if ABSA_PATH.exists():
        absa = pd.read_csv(ABSA_PATH)
        aspect_counts = absa.groupby("review_id")["aspect"].nunique().rename("num_aspects_mentioned")
        df = df.merge(aspect_counts, on="review_id", how="left")
        df["num_aspects_mentioned"] = df["num_aspects_mentioned"].fillna(0)
    else:
        print("(absa_aspect_sentiments.csv not found -- run src/absa/aspect_sentiment.py "
              "first for a richer feature set. Using 0 for num_aspects_mentioned for now.)")
        df["num_aspects_mentioned"] = 0

    df["is_helpful"] = (df["helpful_vote"] >= 1).astype(int)
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    X = df[FEATURE_COLS]
    y = df["is_helpful"]
    print(f"Label distribution:\n{y.value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }

    base_model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
    )

    print("Running RandomizedSearchCV for hyperparameter tuning (15 candidates x 3-fold) ...")
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=15, cv=3,
        scoring="f1", random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best params: {search.best_params_}")

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["less_helpful", "helpful"], digits=3)
    cm = confusion_matrix(y_test, y_pred)

    result_text = (
        "=== XGBoost Review Helpfulness Prediction ===\n"
        f"Label: helpful_vote >= 1 -> 'helpful' ({y.mean():.1%} of reviews), else 'less_helpful'\n"
        f"Best hyperparameters: {search.best_params_}\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision (helpful): {prec:.4f}\n"
        f"Recall (helpful): {rec:.4f}\n"
        f"F1 (helpful): {f1:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=true, cols=pred), label order = less_helpful, helpful:\n{cm}\n"
    )
    print(result_text)
    (METRICS_DIR / "xgboost_helpfulness_results.txt").write_text(result_text)

    joblib.dump(best_model, MODELS_DIR / "xgboost_helpfulness.joblib")
    print(f"Model saved to {MODELS_DIR / 'xgboost_helpfulness.joblib'}")

    importances = best_model.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(FEATURE_COLS)[order], importances[order], color="#4c72b0")
    plt.xlabel("Feature importance")
    plt.title("XGBoost Feature Importance -- Review Helpfulness")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xgboost_feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved feature importance chart to {FIG_DIR / 'xgboost_feature_importance.png'}")


if __name__ == "__main__":
    main()
