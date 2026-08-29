"""
Classical ML sentiment classification: TF-IDF + Logistic Regression and
TF-IDF + Linear SVM. Trained/evaluated on an 80/20 stratified split of the
rating-derived sentiment_label. Trained models + vectorizer are saved to
models/ for reuse (e.g. by the dashboard or RAG pipeline later).
"""
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
METRICS_DIR = BASE_DIR / "reports" / "metrics"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    report = classification_report(y_true, y_pred, labels=LABELS, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    text = (
        f"=== {name} ===\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro F1: {f1_macro:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=true, cols=pred), label order = negative, neutral, positive:\n{cm}\n"
    )
    print(text)
    return text, acc, f1_macro


def main():
    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"].astype(str)
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    all_results = []

    print("Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)
    text, acc_lr, f1_lr = evaluate("TF-IDF + Logistic Regression", y_test, lr_pred)
    all_results.append(text)
    joblib.dump(lr, MODELS_DIR / "tfidf_logreg.joblib")

    print("Training Linear SVM ...")
    svm = LinearSVC(class_weight="balanced")
    svm.fit(X_train_vec, y_train)
    svm_pred = svm.predict(X_test_vec)
    text, acc_svm, f1_svm = evaluate("TF-IDF + Linear SVM", y_test, svm_pred)
    all_results.append(text)
    joblib.dump(svm, MODELS_DIR / "tfidf_svm.joblib")

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    summary = (
        "=== Model Comparison Summary ===\n"
        f"{'Model':<35}{'Accuracy':<12}{'Macro F1':<12}\n"
        f"{'TF-IDF + Logistic Regression':<35}{acc_lr:<12.4f}{f1_lr:<12.4f}\n"
        f"{'TF-IDF + Linear SVM':<35}{acc_svm:<12.4f}{f1_svm:<12.4f}\n"
    )
    print(summary)
    all_results.append(summary)

    (METRICS_DIR / "tfidf_classical_results.txt").write_text("\n".join(all_results))
    print(f"Models saved to {MODELS_DIR}")
    print(f"Results saved to {METRICS_DIR / 'tfidf_classical_results.txt'}")


if __name__ == "__main__":
    main()
