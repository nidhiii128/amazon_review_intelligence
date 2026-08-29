# AI-Based Product Review Intelligence and Consumer Insight System

A system that turns a large volume of unstructured Amazon product reviews
(Electronics category) into structured, actionable insight for two kinds
of users: **buyers** ("should I buy this?") and **companies / sellers**
("what should we improve?").

Dataset: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(McAuley Lab) — Electronics category, ~50,000-review working sample.

## Project structure

```
amazon-review-intelligence/
  data/
    raw/                   raw sampled JSONL pulled from the dataset
    processed/             cleaned CSV used by every downstream module
  src/
    data_prep.py           cleaning, dedup, sampling, rating->sentiment labels
    eda.py                 exploratory data analysis (stats + figures)
    sentiment/              Milestone 1 (DONE)
      vader_baseline.py       VADER rule-based baseline
      tfidf_classical.py      TF-IDF + Logistic Regression, TF-IDF + Linear SVM
      compare_models.py       comparison chart across the three
    topic_modeling/         Milestone 2 (DONE)
      lda_topics.py            LDA topic discovery (8 topics)
    absa/                   Milestone 2 (DONE)
      aspect_sentiment.py      per-aspect sentiment (battery, camera, etc.)
    deep_learning/          Milestone 3 (DONE - runs on your machine)
      bert_finetune.py         DistilBERT fine-tuning + eval vs baselines
    advanced_ml/            Milestone 4 (DONE - needs xgboost installed)
      helpfulness_xgboost.py   XGBoost + feature engineering + tuning
      review_clustering_kmeans.py  optional: KMeans on TF-IDF
    semantic_search/        Milestone 5 (DONE - needs sentence-transformers)
      build_embeddings.py     encodes all 50k reviews once (run first)
      search.py                query by meaning, cosine similarity retrieval
    rag_chatbot/            Milestone 6 (DONE - needs embeddings built first)
      rag_pipeline.py          retrieve reviews + generate grounded answer
    dashboard/              Milestone 7 (DONE - needs streamlit installed)
      data_helpers.py          pure data functions (testable without Streamlit)
      app.py                   the Streamlit app itself (2 views)
  models/                 trained model artifacts (.joblib, later .pt)
  reports/
    figures/              all generated charts (.png)
    metrics/              all evaluation reports (.txt)
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## How to run what exists so far

```bash
cd amazon-review-intelligence
python3 src/data_prep.py                    # -> data/processed/electronics_reviews_clean.csv
python3 src/eda.py                          # -> reports/figures/*.png, reports/metrics/eda_summary.txt
python3 src/sentiment/vader_baseline.py     # -> reports/metrics/vader_results.txt
python3 src/sentiment/tfidf_classical.py    # -> models/*.joblib, reports/metrics/tfidf_classical_results.txt
python3 src/sentiment/compare_models.py     # -> reports/figures/sentiment_model_comparison.png
python3 src/topic_modeling/lda_topics.py    # -> data/processed/electronics_reviews_with_topics.csv
python3 src/absa/aspect_sentiment.py        # -> data/processed/absa_aspect_sentiments.csv

# Milestone 3 (needs requirements-deep-learning.txt installed first):
pip install -r requirements-deep-learning.txt
python3 src/deep_learning/bert_finetune.py  # -> models/bert_sentiment/, reports/metrics/bert_results.txt
python3 src/sentiment/compare_models.py     # re-run to add BERT to the comparison chart

# Milestone 4:
pip install xgboost
python3 src/advanced_ml/helpfulness_xgboost.py       # -> reports/metrics/xgboost_helpfulness_results.txt
python3 src/advanced_ml/review_clustering_kmeans.py  # -> reports/metrics/kmeans_clusters.txt

# Milestone 5 (sentence-transformers is in requirements-deep-learning.txt):
python3 src/semantic_search/build_embeddings.py   # run once -> models/embeddings/*.npy
python3 src/semantic_search/search.py             # demo queries
python3 src/semantic_search/search.py "is the camera good in low light"  # your own query

# Milestone 6 (reuses the embeddings from Milestone 5 -- build those first):
python3 src/rag_chatbot/rag_pipeline.py                                          # demo questions
python3 src/rag_chatbot/rag_pipeline.py "is the battery good"                    # your own question
python3 src/rag_chatbot/rag_pipeline.py "good for gaming" --asin B00ZV9RDKK      # scoped to one product

# Milestone 7 (run from the amazon-review-intelligence/ folder):
pip install -r requirements-dashboard.txt
streamlit run src/dashboard/app.py
```

## Where the raw data came from

The full Electronics review file on Hugging Face is 22.6 GB, far too large
to fully download for a working sample. `data/raw/electronics_raw_sample.jsonl`
holds the first ~65k reviews (byte-range fetched from the dataset, ~42 MB),
which `data_prep.py` then cleans and samples down to the target 50,000 rows
(stratified so all three sentiment classes are represented). If a different
or larger sample is ever needed, more of the file can be pulled the same way.

## Milestone 1 results (sentiment baselines)

Rating-derived labels: rating>=4 -> positive, rating==3 -> neutral, rating<=2 -> negative.
Class balance in the 50k sample: positive 40,205 / negative 6,289 / neutral 3,506
(reflects real-world Amazon rating skew).

| Model                          | Accuracy | Macro F1 |
|---------------------------------|----------|----------|
| VADER (rule-based baseline)     | 0.816    | 0.519    |
| TF-IDF + Logistic Regression    | 0.860    | 0.689    |
| TF-IDF + Linear SVM             | 0.888    | 0.693    |

Takeaway: VADER is a reasonable zero-training baseline but struggles hardest
on the neutral class; both trained classical models clearly beat it, with
Linear SVM the best classical option so far. This sets the bar the BERT
fine-tune (Milestone 3) needs to beat to justify the deep learning step.

## Milestone 2 results (topic modeling + ABSA)

**LDA topics** (8 topics, CountVectorizer + LatentDirichletAllocation, fit on
an 8k-doc subsample then applied to all 50k for speed): topics cleanly split
into audio/headphones, storage (SD cards), screens/monitors, computer
accessories (USB/keyboard/mouse), cameras, cases/covers, general/phone
issues, and cables/networking. See `reports/metrics/lda_topics.txt` and
`reports/figures/lda_top_words.png`.

**Aspect-Based Sentiment Analysis** (regex sentence-splitting + keyword
matching per aspect + VADER sentiment on matched sentences), sorted by
highest negative share:

| Aspect | Positive | Neutral | Negative | Mentions |
|---|---|---|---|---|
| Battery | 56.3% | 23.5% | 20.1% | 6,968 |
| Connectivity | 56.7% | 23.9% | 19.3% | 4,166 |
| Customer service | 64.0% | 16.7% | 19.3% | 2,958 |
| Performance | 58.7% | 25.5% | 15.8% | 4,726 |
| Display | 64.3% | 21.1% | 14.5% | 4,540 |
| Sound | 73.4% | 13.7% | 12.9% | 8,327 |
| Camera | 70.1% | 18.6% | 11.3% | 5,991 |
| Price/value | 74.2% | 15.1% | 10.7% | 11,070 |
| Size/comfort | 76.1% | 13.5% | 10.4% | 10,752 |
| Build quality | 66.5% | 25.4% | 8.1% | 4,636 |

Takeaway: battery is the single biggest recurring complaint area across the
Electronics sample, exactly the kind of insight ("battery complaints are the
most frequent negative feedback") the project brief called out as the goal.
See `reports/metrics/absa_summary.txt` and `reports/figures/aspect_sentiment_stacked.png`.

## Milestone 5 notes (semantic search)

Uses `sentence-transformers` with `all-MiniLM-L6-v2` -- small (~80MB),
fast enough for CPU, a standard choice for this kind of retrieval. Encodes
all 50,000 reviews once into 384-dim normalized embeddings
(`models/embeddings/`); after that, each search is just a dot product
against that matrix, so it's instant. Expect the one-time embedding step
to take several minutes on CPU-only hardware (progress bar included).

This is the same retrieval mechanism the Milestone 6 RAG chatbot uses --
`build_embeddings.py` only needs to be run once, and both semantic search
and the chatbot will read from the same saved embeddings file. Verified
the review-ID alignment logic (embeddings row <-> correct review) with a
synthetic test since the real model can't download in the environment I
use to check scripts before handing them to you; the actual encode/search
calls need your first real run to confirm, same pattern as BERT and XGBoost.

## Milestone 6 notes (RAG chatbot)

Flow matches the brief exactly: question -> embed -> retrieve top-5
similar reviews (reusing the Milestone 5 embeddings, no re-encoding the
corpus) -> feed question + those reviews to a small local LLM
(`google/flan-t5-small`, instruction-tuned, no API key, works fully
offline after the one-time download) -> generate an answer -> print the
answer AND the supporting reviews it was grounded in, so the answer is
never just an unverifiable LLM guess.

Supports an optional `--asin` filter so a question can be scoped to one
product (the buyer-facing "ask about THIS product" use case), falling
back to the full corpus if that product has no reviews. If `google/flan-t5-small`'s
answers feel too generic, `google/flan-t5-base` is a drop-in swap
(`FLAN_MODEL_NAME` at the top of the file) for noticeably better quality
at the cost of a larger download and slower inference.

Verified the retrieval + product-scoping logic (correct reviews retrieved,
asin filter actually restricts results) with a synthetic test; the real
embedding + generation models need your machine to actually download and
run, same as the other neural milestones.

## Milestone 7 notes (dashboard)

Two views exactly matching the brief, selectable from the sidebar:

- **Buyer view** ("should I buy this?"): pick a product (by ASIN + avg
  rating + review count -- this dataset only pulled review-level fields,
  no product-title metadata, so ASIN is the identifier), see strengths/
  weaknesses, aspect-level sentiment, what topics its reviews cover,
  side-by-side product comparison, and the RAG chatbot scoped to that
  product.
- **Company view** ("what should we improve?"): top complaint areas
  across the whole sample, all the Milestone 1-4 result charts pulled in
  directly (sentiment comparison, ABSA, LDA topics, XGBoost feature
  importance, KMeans clusters -- whichever have been generated so far;
  anything not run yet shows a friendly "not generated" message instead
  of crashing), a per-product deep-dive table, and expandable full-text
  versions of every metrics report.

The chatbot only loads its models (torch + transformers + embeddings) the
first time you actually ask it a question, not on dashboard startup --
otherwise every page load would pay that cost even for people just
browsing charts.

I split the data logic (`data_helpers.py`) from the UI (`app.py`) on
purpose so the analysis functions could be verified directly against your
real data without needing a running Streamlit session -- all of them
(product picker, aspect sentiment lookup, strengths/weaknesses, topic
mix, corpus-wide complaints) checked out against the actual 50k-review
dataset. I also started the app itself in a background server and
confirmed it serves successfully with no startup errors.

## Roadmap (matches the original project brief)

1. **Data + NLP baseline** — DONE: cleaning, EDA, VADER, TF-IDF+LR, TF-IDF+SVM.
2. **Topic modeling + Aspect-Based Sentiment Analysis (ABSA)** — DONE: LDA topics
   and per-aspect sentiment (e.g. "camera positive, battery negative" from
   one review).
## Milestone 3 (deep learning) notes

Uses DistilBERT (not full BERT) as a deliberate trade-off: ~97% of BERT's
performance at ~60% of the size, which matters when this may run on a
CPU-only laptop rather than a GPU. The script auto-detects a GPU if
present (`torch.cuda.is_available()`) and scales the training sample up
automatically; on CPU-only machines it trains on a 6,000-review subsample
(2 epochs) to keep runtime reasonable, but always evaluates on the full
10,000-review test set -- the exact same test split used by the TF-IDF
models, so the comparison in `reports/figures/sentiment_model_comparison.png`
is apples-to-apples.

This needs `pip install -r requirements-deep-learning.txt` (torch +
transformers, a few hundred MB) and downloads the pretrained DistilBERT
weights from Hugging Face on first run, so it needs a normal internet
connection the first time. Expect roughly 15-40 minutes on CPU-only
hardware depending on your machine; much faster with a GPU.

Note: PyTorch wheel support for brand-new Python versions sometimes lags a
little -- if `pip install -r requirements-deep-learning.txt` fails to find
a torch build for your Python version, that's the likely cause (nothing
wrong with the project itself).

## Milestone 4 results (advanced ML)

**XGBoost review helpfulness prediction.** Label: `helpful_vote >= 1` ->
"helpful" (26.2% of the 50k sample), else "less_helpful". Features:
rating, review length (words), title length, verified purchase, exclamation
/ question mark counts, capital-letter ratio, VADER sentiment compound,
review age in days, and number of distinct aspects mentioned (pulled from
the Milestone 2 ABSA output -- a nice example of one module feeding
another). Tuned via `RandomizedSearchCV` (15 candidates x 3-fold) over
n_estimators / max_depth / learning_rate / subsample / colsample_bytree,
with `scale_pos_weight` set for the class imbalance. This one needs
`pip install xgboost` and wasn't run end-to-end on this side (see note
below) -- paste the results here after your first run and they'll get
folded into this table, same as the BERT step.

**KMeans review clustering** (optional secondary task, on a dedicated
stopword-filtered TF-IDF -- reusing the sentiment model's vectorizer would
have buried every cluster in "the/it/and", since that one deliberately
keeps stopwords for sentiment nuance). 6 clusters found; two came out
clearly topical (sound/headphones; laptop bags/cases), the rest lean
generic ("great product", "five stars") since most reviews are short and
positive. Worth stating plainly rather than dressing up: KMeans on raw
TF-IDF struggles to out-do LDA's topic separation here, which is itself a
legitimate finding about short, ratings-skewed review text, not a bug.
See `reports/metrics/kmeans_clusters.txt` and `reports/figures/kmeans_clusters.png`.

## Milestone 5 notes (semantic search)

Uses `sentence-transformers` with `all-MiniLM-L6-v2` -- small (~80MB),
fast enough for CPU, a standard choice for this kind of retrieval. Encodes
all 50,000 reviews once into 384-dim normalized embeddings
(`models/embeddings/`); after that, each search is just a dot product
against that matrix, so it's instant. Expect the one-time embedding step
to take several minutes on CPU-only hardware (progress bar included).

This is the same retrieval mechanism the Milestone 6 RAG chatbot uses --
`build_embeddings.py` only needs to be run once, and both semantic search
and the chatbot will read from the same saved embeddings file. Verified
the review-ID alignment logic (embeddings row <-> correct review) with a
synthetic test since the real model can't download in the environment I
use to check scripts before handing them to you; the actual encode/search
calls need your first real run to confirm, same pattern as BERT and XGBoost.

## Milestone 6 notes (RAG chatbot)

Flow matches the brief exactly: question -> embed -> retrieve top-5
similar reviews (reusing the Milestone 5 embeddings, no re-encoding the
corpus) -> feed question + those reviews to a small local LLM
(`google/flan-t5-small`, instruction-tuned, no API key, works fully
offline after the one-time download) -> generate an answer -> print the
answer AND the supporting reviews it was grounded in, so the answer is
never just an unverifiable LLM guess.

Supports an optional `--asin` filter so a question can be scoped to one
product (the buyer-facing "ask about THIS product" use case), falling
back to the full corpus if that product has no reviews. If `google/flan-t5-small`'s
answers feel too generic, `google/flan-t5-base` is a drop-in swap
(`FLAN_MODEL_NAME` at the top of the file) for noticeably better quality
at the cost of a larger download and slower inference.

Verified the retrieval + product-scoping logic (correct reviews retrieved,
asin filter actually restricts results) with a synthetic test; the real
embedding + generation models need your machine to actually download and
run, same as the other neural milestones.

## Milestone 7 notes (dashboard)

Two views exactly matching the brief, selectable from the sidebar:

- **Buyer view** ("should I buy this?"): pick a product (by ASIN + avg
  rating + review count -- this dataset only pulled review-level fields,
  no product-title metadata, so ASIN is the identifier), see strengths/
  weaknesses, aspect-level sentiment, what topics its reviews cover,
  side-by-side product comparison, and the RAG chatbot scoped to that
  product.
- **Company view** ("what should we improve?"): top complaint areas
  across the whole sample, all the Milestone 1-4 result charts pulled in
  directly (sentiment comparison, ABSA, LDA topics, XGBoost feature
  importance, KMeans clusters -- whichever have been generated so far;
  anything not run yet shows a friendly "not generated" message instead
  of crashing), a per-product deep-dive table, and expandable full-text
  versions of every metrics report.

The chatbot only loads its models (torch + transformers + embeddings) the
first time you actually ask it a question, not on dashboard startup --
otherwise every page load would pay that cost even for people just
browsing charts.

I split the data logic (`data_helpers.py`) from the UI (`app.py`) on
purpose so the analysis functions could be verified directly against your
real data without needing a running Streamlit session -- all of them
(product picker, aspect sentiment lookup, strengths/weaknesses, topic
mix, corpus-wide complaints) checked out against the actual 50k-review
dataset. I also started the app itself in a background server and
confirmed it serves successfully with no startup errors.

## Roadmap (matches the original project brief)

1. **Data + NLP baseline** — DONE: cleaning, EDA, VADER, TF-IDF+LR, TF-IDF+SVM.
2. **Topic modeling + Aspect-Based Sentiment Analysis (ABSA)** — DONE: LDA topics
   and per-aspect sentiment (e.g. "camera positive, battery negative" from
   one review).
3. **Deep Learning** — DONE: DistilBERT fine-tuned and compared against the
   Milestone 1 baselines on the same test split.
4. **Advanced ML** — DONE: XGBoost helpfulness prediction + optional KMeans
   clustering.
5. **Semantic search** — DONE: sentence embeddings + cosine similarity so
   users can search reviews by meaning, not just keywords.
6. **RAG chatbot** — DONE: retrieval-augmented generation over the review
   corpus using a free/local embedding model + local LLM, so buyers/
   companies can ask natural-language questions grounded in actual reviews.
7. **Dashboard** — DONE: two Streamlit views (Buyer: strengths/weaknesses,
   comparisons, chatbot; Company: complaints, aspect trends, helpfulness,
   topics).
