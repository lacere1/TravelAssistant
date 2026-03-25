"""
evaluate_classifier.py — Intent classifier evaluation script.

Usage:
    python evaluate_classifier.py               # CV + latency for SBERT and TF-IDF

    pip install --upgrade transformers huggingface_hub
    python evaluate_classifier.py --zero-shot   # also benchmark zero-shot (slow, ~600ms/query)
"""

import os
import sys
import time
import argparse
import numpy as np

# Fix Windows HuggingFace cache bug — ETag values contain '"' which is
# an invalid character in Windows filenames, causing a lock file error.
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.path.expanduser("~"), "hf_cache"))

sys.path.insert(0, ".")

from training_data import INTENT_TRAINING_DATA


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    texts, labels = [], []
    for intent, examples in INTENT_TRAINING_DATA.items():
        for ex in examples:
            texts.append(ex)
            labels.append(intent)
    return texts, labels


# ── SBERT + LogReg ────────────────────────────────────────────────────────────

def run_sbert_cv(texts, labels, n_splits=5):
    
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    print("[SBERT] Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[SBERT] Encoding {len(texts)} examples...")
    embeddings = model.encode(texts, show_progress_bar=False)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(
        max_iter=1000, multi_class="multinomial", solver="lbfgs", C=10.0
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, y, cv=skf, scoring="accuracy")

    # Latency: mean over 100 single-query inferences
    clf.fit(embeddings, y)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        emb = model.encode(["bus times at Wembley"], show_progress_bar=False)
        clf.predict_proba(emb)
        latencies.append((time.perf_counter() - t0) * 1000)

    return scores, np.mean(latencies)


# ── TF-IDF + LogReg (baseline) ────────────────────────────────────────────────

def run_tfidf_cv(texts, labels, n_splits=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs", C=10.0
        )),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, texts, y, cv=skf, scoring="accuracy")

    # Latency
    pipe.fit(texts, y)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        pipe.predict_proba(["bus times at Wembley"])
        latencies.append((time.perf_counter() - t0) * 1000)

    return scores, np.mean(latencies)


# ── Zero-shot pipeline (optional, slow) ───────────────────────────────────────

def run_zero_shot_sample(texts, labels, n_samples=20):
    """
    Runs zero-shot classification on a random sample.
    Full dataset would take ~2 minutes; 20 examples gives a representative latency.
    """
    import random
    from transformers import pipeline

    # Map natural-language candidate labels back to intent keys
    candidate_labels = [
        "greeting",
        "goodbye",
        "bus or train timetable",
        "transit disruption or service status",
        "journey planning or route finding",
    ]
    reverse_map = {
        "greeting":                               "greeting",
        "goodbye":                                "goodbye",
        "bus or train timetable":                 "ask_timetable",
        "transit disruption or service status":   "ask_transit_disruption",
        "journey planning or route finding":      "journey_planning",
    }

    random.seed(42)
    indices = random.sample(range(len(texts)), n_samples)
    sample_texts  = [texts[i]  for i in indices]
    sample_labels = [labels[i] for i in indices]

    print("[Zero-shot] Loading facebook/bart-large-mnli (first run downloads ~1.6 GB)...")
    zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    correct, latencies = 0, []
    for text, true_label in zip(sample_texts, sample_labels):
        t0 = time.perf_counter()
        result = zs(text, candidate_labels=candidate_labels)
        latencies.append((time.perf_counter() - t0) * 1000)
        if reverse_map[result["labels"][0]] == true_label:
            correct += 1

    return correct / n_samples, np.mean(latencies), n_samples


# ── Pretty-print table ────────────────────────────────────────────────────────

def print_table(rows, headers):
    col_w = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for row in rows:
        print(fmt.format(*row))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zero-shot", action="store_true",
        help="Also benchmark zero-shot pipeline (slow — downloads ~1.6 GB on first run)"
    )
    args = parser.parse_args()

    texts, labels = build_dataset()
    print(f"Dataset: {len(texts)} examples across {len(set(labels))} classes")
    for intent, examples in INTENT_TRAINING_DATA.items():
        print(f"  {intent}: {len(examples)} examples")
    print()

    # ── SBERT ──
    print("=" * 60)
    print("1/2  Sentence-BERT + LogisticRegression  (production model)")
    print("=" * 60)
    sbert_scores, sbert_latency = run_sbert_cv(texts, labels)
    print(f"Fold accuracies : {[f'{s:.4f}' for s in sbert_scores]}")
    print(f"Mean accuracy   : {np.mean(sbert_scores)*100:.2f}%")
    print(f"Std             : ±{np.std(sbert_scores)*100:.2f}%")
    print(f"Mean latency    : {sbert_latency:.1f} ms  (100-query average)\n")

    # ── TF-IDF ──
    print("=" * 60)
    print("2/2  TF-IDF + LogisticRegression  (baseline)")
    print("=" * 60)
    tfidf_scores, tfidf_latency = run_tfidf_cv(texts, labels)
    print(f"Fold accuracies : {[f'{s:.4f}' for s in tfidf_scores]}")
    print(f"Mean accuracy   : {np.mean(tfidf_scores)*100:.2f}%")
    print(f"Std             : ±{np.std(tfidf_scores)*100:.2f}%")
    print(f"Mean latency    : {tfidf_latency:.1f} ms  (100-query average)\n")

    rows = [
        [
            "Sentence-BERT + LogReg (production)",
            f"{np.mean(sbert_scores)*100:.1f}%",
            f"±{np.std(sbert_scores)*100:.1f}pp",
            f"{sbert_latency:.1f} ms",
        ],
        [
            "TF-IDF + LogReg (baseline)",
            f"{np.mean(tfidf_scores)*100:.1f}%",
            f"±{np.std(tfidf_scores)*100:.1f}pp",
            f"{tfidf_latency:.1f} ms",
        ],
    ]

    # ── Zero-shot (optional) ──
    if args.zero_shot:
        print("=" * 60)
        print("3/3  Zero-shot pipeline  (HuggingFace bart-large-mnli)")
        print("=" * 60)
        zs_acc, zs_latency, zs_n = run_zero_shot_sample(texts, labels)
        print(f"Accuracy (n={zs_n}) : {zs_acc*100:.1f}%")
        print(f"Mean latency       : {zs_latency:.1f} ms\n")
        rows.append([
            f"Zero-shot bart-large-mnli (n={zs_n})",
            f"{zs_acc*100:.1f}%",
            "—",
            f"{zs_latency:.1f} ms",
        ])

    # ── Summary ──
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print_table(rows, ["Approach", "CV Accuracy", "Std", "Latency"])
    print()

    improvement = (np.mean(sbert_scores) - np.mean(tfidf_scores)) * 100
    print(f"SBERT improvement over TF-IDF baseline : +{improvement:.1f} percentage points")

    if args.zero_shot:
        ratio = zs_latency / sbert_latency
        print(f"Zero-shot is {ratio:.1f}x slower than SBERT per query")
