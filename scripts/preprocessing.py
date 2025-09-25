# scripts/preprocessing.py
"""
Preprocessing & EDA for BBC dataset.
- Reads: dataset/bbc_news_text_complexity_summarization.csv
- Saves plots to: plots/
"""

from pathlib import Path
import os
import re
import string

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# -----------------------------
# Paths (robust to CWD)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = PROJECT_ROOT / "dataset" / "bbc_news_text_complexity_summarization.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Setup
# -----------------------------
sns.set(style="whitegrid")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
NLTK_STOP = set(stopwords.words("english"))

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Helpers
# -----------------------------
def savefig(fig, name: str):
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📸 Saved: {out}")

def get_top_bigrams(texts, topn=10):
    """Return top-N bigrams and counts from iterable of strings."""
    vec = CountVectorizer(ngram_range=(2, 2))
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array([t for t, _ in sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])])
    # Sort by frequency descending
    order = np.argsort(-sums)
    bigrams = vocab[order][:topn]
    counts = sums[order][:topn]
    return list(zip(bigrams, counts))

def clean_text(text: str) -> str:
    """Basic cleaning + stopword removal + lemmatization."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # spaces instead of punctuation to keep token boundaries
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # NLTK stopwords
    tokens = [w for w in text.split() if w not in NLTK_STOP and len(w) > 2]
    text = " ".join(tokens)

    # spaCy lemmatization + spaCy stopwords
    doc = nlp(text)
    lemmas = [
        tok.lemma_
        for tok in doc
        if tok.text not in STOP_WORDS and not tok.is_punct and not tok.is_space
    ]
    return " ".join(lemmas)


# -----------------------------
# Load data
# -----------------------------
if not DATA_CSV.exists():
    raise FileNotFoundError(
        f"Cannot find data at {DATA_CSV}. "
        "Run `python scripts/download_data.py` first to populate dataset/."
    )

data = pd.read_csv(DATA_CSV)
# Basic sanity
if "text" not in data.columns or "labels" not in data.columns:
    raise ValueError("Expected columns `text` and `labels` not found in the CSV.")

data["text"] = data["text"].fillna("")

print(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns.")
print("Label distribution:\n", data["labels"].value_counts())

# -----------------------------
# Plot 1: Class counts
# -----------------------------
article_count = (
    data.groupby("labels")["labels"].count()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)

fig = plt.figure(figsize=(12, 6))
sns.barplot(data=article_count, x="labels", y="count", palette="Set2")
plt.xlabel("Article class", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.title("Article Classes in BBC News", fontsize=20)
plt.tight_layout()
savefig(fig, "class_counts.png")

# -----------------------------
# Aggregations (sentences/words/readability)
# -----------------------------
if "no_sentences" in data.columns:
    avg_no_of_sentences = (
        data.groupby("labels")["no_sentences"].mean()
        .reset_index(name="avg_no_of_sentences")
        .sort_values(by="avg_no_of_sentences", ascending=False)
    )
else:
    avg_no_of_sentences = None

data["word_count"] = data["text"].str.split().map(lambda x: len(x))

avg_no_of_words = (
    data.groupby("labels")["word_count"].mean()
    .reset_index(name="avg_no_of_words")
    .sort_values(by="avg_no_of_words", ascending=False)
)

col_flesch = "Flesch Reading Ease Score"
col_dale = "Dale-Chall Readability Score"
avg_flesch = (
    data.groupby("labels")[col_flesch].mean().reset_index(name="avg_flesch").sort_values(
        by="avg_flesch", ascending=False
    )
) if col_flesch in data.columns else None

avg_dale = (
    data.groupby("labels")[col_dale].mean().reset_index(name="avg_dale").sort_values(
        by="avg_dale", ascending=False
    )
) if col_dale in data.columns else None

# -----------------------------
# Plot 2: Avg sentences & words
# -----------------------------
if avg_no_of_sentences is not None:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
    sns.barplot(data=avg_no_of_sentences, x="labels", y="avg_no_of_sentences", ax=ax1, palette="Set3")
    ax1.set_title("Average Number of Sentences by Class", fontsize=15)
    ax1.set_xlabel("Class"); ax1.set_ylabel("Avg sentences")

    sns.barplot(data=avg_no_of_words, x="labels", y="avg_no_of_words", ax=ax2, palette="Set3")
    ax2.set_title("Average Number of Words by Class", fontsize=15)
    ax2.set_xlabel("Class"); ax2.set_ylabel("Avg words")

    plt.tight_layout()
    savefig(fig, "avg_sentences_words.png")
else:
    # Only words available
    fig = plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_no_of_words, x="labels", y="avg_no_of_words", palette="Set3")
    plt.title("Average Number of Words by Class", fontsize=15)
    plt.xlabel("Class"); plt.ylabel("Avg words")
    plt.tight_layout()
    savefig(fig, "avg_words_only.png")

# -----------------------------
# Plot 3: Readability (Flesch & Dale-Chall)
# -----------------------------
if avg_flesch is not None and avg_dale is not None:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

    sns.barplot(data=avg_flesch, x="labels", y="avg_flesch", ax=ax1, palette="Pastel1")
    ax1.set_title("Average Flesch Reading Ease Score by Class", fontsize=14)
    ax1.axhline(y=50, color="purple", linestyle="--", label="Fairly Difficult")
    ax1.axhline(y=60, color="black", linestyle="--", label="Standard")
    ax1.axhline(y=70, color="blue", linestyle="--", label="Fairly Easy")
    ax1.legend(loc="lower right", title="Flesch Guide")
    ax1.set_xlabel("Class"); ax1.set_ylabel("Avg Flesch")

    sns.barplot(data=avg_dale, x="labels", y="avg_dale", ax=ax2, palette="Pastel1")
    ax2.set_title("Average Dale-Chall Readability Score by Class", fontsize=14)
    ax2.axhline(y=7, color="purple", linestyle="--", label="Avg 9–10th grade")
    ax2.axhline(y=8, color="black", linestyle="--", label="Avg 11–12th grade")
    ax2.axhline(y=9, color="blue", linestyle="--", label="Avg college (13–15th)")
    ax2.legend(loc="lower right", title="Dale-Chall Guide")
    ax2.set_xlabel("Class"); ax2.set_ylabel("Avg Dale-Chall")

    plt.tight_layout()
    savefig(fig, "avg_readability.png")

# -----------------------------
# Top bigrams (raw text)
# -----------------------------
top_bigrams_raw = get_top_bigrams(data["text"], topn=10)
bx, by = map(list, zip(*top_bigrams_raw))

fig = plt.figure(figsize=(10, 6))
sns.barplot(x=by, y=bx, color="steelblue")
plt.xlabel("Count", size=12); plt.ylabel("Bigrams", size=12)
plt.title("Top Bigrams in BBC Articles (Raw Text)", size=16)
plt.tight_layout()
savefig(fig, "top_bigrams_raw.png")

# -----------------------------
# Clean text column
# -----------------------------
data["clean_text"] = data["text"].apply(clean_text)

# Optionally drop super-common tokens you don't want
COMMON_WORDS = {"said"}
data["clean_text"] = data["clean_text"].apply(
    lambda s: " ".join(w for w in s.split() if w not in COMMON_WORDS)
)

# -----------------------------
# WordClouds
# -----------------------------
# Overall
wc_all = WordCloud(width=1000, height=600).generate(" ".join(data["clean_text"]))

# Per class (guard if some labels missing)
labels_unique = sorted(data["labels"].unique().tolist())
wcs = {}
for lbl in labels_unique:
    txt = " ".join(data.loc[data["labels"] == lbl, "clean_text"])
    # set different background colors for variety
    wcs[lbl] = WordCloud(width=800, height=500, background_color="white").generate(txt)

# Grid: up to 6 panels (overall + first 5 classes)
cols = 3
rows = 2
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
axes = axes.ravel()

axes[0].imshow(wc_all); axes[0].axis("off"); axes[0].set_title("Wordcloud (All)", fontsize=12)

i = 1
for lbl in labels_unique[: (rows * cols - 1)]:
    axes[i].imshow(wcs[lbl])
    axes[i].axis("off")
    axes[i].set_title(f"Wordcloud ({lbl})", fontsize=12)
    i += 1

# Hide any unused axes
for j in range(i, rows * cols):
    axes[j].axis("off")

plt.tight_layout()
savefig(fig, "wordclouds_grid.png")

# -----------------------------
# Top bigrams (clean text)
# -----------------------------
top_bigrams_clean = get_top_bigrams(data["clean_text"], topn=10)
bx, by = map(list, zip(*top_bigrams_clean))

fig = plt.figure(figsize=(10, 6))
sns.barplot(x=by, y=bx, color="steelblue")
plt.xlabel("Count", size=12); plt.ylabel("Bigrams", size=12)
plt.title("Top Bigrams in BBC Articles (Preprocessed Text)", size=16)
plt.tight_layout()
savefig(fig, "top_bigrams_clean.png")

print("✅ Preprocessing & EDA complete.")
