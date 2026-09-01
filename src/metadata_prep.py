"""
Product metadata integration.

Reviews only carry `parent_asin` -- no product title, image, or brand.
The same McAuley-Lab Amazon-Reviews-2023 dataset also publishes per-category
product metadata (title, store, images, category, etc.), keyed by the same
`parent_asin`. This script streams that metadata file, keeps only the rows
matching a `parent_asin` already present in our 50k-review sample, and
writes a small `products.csv` -- the reviews stay untouched; the dashboard
joins the two on `parent_asin` at load time instead of merging the huge
metadata file into every review row.

Step 1: stream raw/meta_categories/meta_Electronics.jsonl (~5.2 GB, not
gzipped) from Hugging Face, filtering as we go -> data/raw/
meta_electronics_sample.jsonl (only the ~34k parent_asins we actually need).
Step 2: clean that into data/processed/products.csv.

Needs `requests` (already installed). No `datasets` library dependency.

Usage:
  python3 src/metadata_prep.py
"""
import json
import pathlib
import sys
import time

import pandas as pd
import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
REVIEWS_PATH = BASE_DIR / "data" / "processed" / "electronics_reviews_clean.csv"
RAW_META_PATH = BASE_DIR / "data" / "raw" / "meta_electronics_sample.jsonl"
PRODUCTS_PATH = BASE_DIR / "data" / "processed" / "products.csv"

META_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_Electronics.jsonl"
)


def fetch_matching_metadata(target_parent_asins: set) -> int:
    """Stream the metadata file, writing only rows whose parent_asin is in
    target_parent_asins. Stops early once every target has been found."""
    RAW_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    found = {}
    t0 = time.time()
    bytes_seen = 0

    print(f"Streaming {META_URL} ...")
    print(f"Looking for {len(target_parent_asins):,} distinct parent_asins ...")
    with requests.get(META_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        buffer = ""
        for chunk in resp.iter_content(chunk_size=1 << 20, decode_unicode=False):
            bytes_seen += len(chunk)
            buffer += chunk.decode("utf-8", errors="ignore")
            *lines, buffer = buffer.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pa = obj.get("parent_asin")
                if pa in target_parent_asins and pa not in found:
                    found[pa] = obj

            if bytes_seen % (200 << 20) < (1 << 20):  # ~every 200MB
                pct = bytes_seen / total_size * 100 if total_size else 0
                elapsed = time.time() - t0
                print(f"  scanned {bytes_seen / 1e6:,.0f} MB ({pct:.1f}%), "
                      f"matched {len(found):,}/{len(target_parent_asins):,}, "
                      f"{elapsed:.0f}s elapsed", flush=True)

            if len(found) == len(target_parent_asins):
                print("All target parent_asins found -- stopping early.")
                break

    elapsed = time.time() - t0
    print(f"\nDone: matched {len(found):,}/{len(target_parent_asins):,} "
          f"parent_asins in {elapsed / 60:.1f} minutes ({bytes_seen / 1e6:,.0f} MB read)")

    with open(RAW_META_PATH, "w", encoding="utf-8") as f:
        for obj in found.values():
            f.write(json.dumps(obj) + "\n")
    print(f"Saved raw matched metadata to {RAW_META_PATH}")
    return len(found)


def build_products_csv():
    rows = []
    with open(RAW_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            image_url = None
            images = obj.get("images") or []
            if images:
                first = images[0]
                image_url = first.get("large_image_url") or first.get("hi_res") or first.get("thumb")

            rows.append({
                "parent_asin": obj.get("parent_asin"),
                "product_name": (obj.get("title") or "").strip(),
                "brand": (obj.get("store") or "").strip(),
                "image_url": image_url,
                "main_category": obj.get("main_category"),
                "average_rating": obj.get("average_rating"),
                "rating_number": obj.get("rating_number"),
            })

    df = pd.DataFrame(rows)
    df = df[df["product_name"] != ""]
    df = df.drop_duplicates(subset=["parent_asin"])
    PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRODUCTS_PATH, index=False)
    print(f"\nSaved {len(df):,} products to {PRODUCTS_PATH}")
    return df


def main():
    reviews = pd.read_csv(REVIEWS_PATH, usecols=["parent_asin"])
    target = set(reviews["parent_asin"].dropna().unique())

    if RAW_META_PATH.exists():
        print(f"{RAW_META_PATH} already exists -- skipping download, rebuilding products.csv from it.")
        print("(delete it first if you want to re-fetch from Hugging Face.)")
    else:
        fetch_matching_metadata(target)

    df = build_products_csv()

    join_rate = df["parent_asin"].nunique() / len(target) * 100
    print(f"\nMetadata join rate: {join_rate:.1f}% of parent_asins in the review "
          f"sample have product metadata ({df['parent_asin'].nunique():,}/{len(target):,}).")
    print("Products without metadata fall back to a generic 'Product {parent_asin}' "
          "display in the dashboard.")


if __name__ == "__main__":
    main()
