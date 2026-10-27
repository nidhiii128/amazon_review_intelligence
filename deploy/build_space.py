"""Assemble a self-contained Hugging Face Space (Docker) from this repo.

    python deploy/build_space.py          # writes deploy/hf_space/

Only what the dashboard needs at run time is copied: the app code, the
processed tables, the review embeddings and the helpfulness model. The two
language models (MiniLM, FLAN-T5) are downloaded into the image at build time.
"""
import pathlib
import shutil

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "deploy" / "hf_space"

CODE = [
    "src/dashboard/app.py", "src/dashboard/data_helpers.py", "src/dashboard/theme.py", "src/dashboard/widgets.py",
    "src/dashboard/sections", "src/rag_chatbot/rag_pipeline.py", "src/advanced_ml/helpfulness_xgboost.py",
    ".streamlit/config.toml",
]
DATA = [
    "data/processed/electronics_reviews_clean.csv", "data/processed/absa_aspect_sentiments.csv",
    "data/processed/products.csv", "data/processed/suspicious_reviews.csv",
    "data/processed/lstm_predictions.csv", "data/processed/helpfulness_scores.csv",
    "models/embeddings/review_embeddings.npy", "models/embeddings/review_ids.npy",
    "models/xgboost_helpfulness.joblib",
]

README = """---
title: ReviewIQ
emoji: 🛒
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ReviewIQ

Product review intelligence for Amazon Electronics: search a product, see what
reviewers say by aspect, check how far to trust the reviews, and ask questions
answered only from that product's reviews.

Vidyalankar Institute of Technology. Team: Nidhi Bhawari, Aarti Sakpal,
Gayatri Patil, Prachi Wakode. Guide: Prof. Sneha Annappanavar.

Data: Amazon Reviews 2023 (Electronics), a 50,000 review sample.
"""

REQUIREMENTS = """streamlit==1.62.0
pandas==3.0.5
numpy==2.5.2
scikit-learn==1.9.0
xgboost==3.4.1
joblib==1.6.0
vaderSentiment==3.3.2
sentence-transformers==6.0.1
transformers==5.16.1
"""

DOCKERFILE = """FROM python:3.12-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \\
 && pip install --no-cache-dir -r requirements.txt
USER user
# bake the two language models into the image so the first question is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); \\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \\
AutoTokenizer.from_pretrained('google/flan-t5-base'); AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"
COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
"""


def copy(rel):
    src, dst = BASE / rel, OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        shutil.copy2(src, dst)


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for rel in CODE + DATA:
    copy(rel)
(OUT / "README.md").write_text(README)
(OUT / "requirements.txt").write_text(REQUIREMENTS)
(OUT / "Dockerfile").write_text(DOCKERFILE)
(OUT / ".dockerignore").write_text("__pycache__/\n*.pyc\n")
total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
print(f"built {OUT} ({total:.0f} MB)")
