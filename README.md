# NewsCLF — BBC Topic Classifier 

**NewsCLF** is a  text classifier that predicts the main topic of a news article as one of: **business · entertainment · politics · sport · tech**.
It ships as a clean **Python package** (importable API) and a single **CLI** (`newsclf`) for training and inference. Runs on **CPU**, **Apple Silicon (MPS)**, or **CUDA** automatically.

---

## Goals

* Deliver a **robust, reproducible** BBC-style news classifier with strong accuracy.
* Provide a **library API** for apps/services and a **CLI** for quick ops.
* Save **artifacts, metrics, reports, and confusion matrices** per run for auditability.

---

## Tech

* **PyTorch**, **Hugging Face Transformers** (default: `distilbert-base-uncased`)
* **scikit-learn**, **pandas**, **numpy**
* **matplotlib** 

---

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

> Expected data file: `dataset/bbc_news_text_complexity_summarization.csv` with columns: `text`, `labels`.

---

## Train

```bash
newsclf train \
  --data_csv dataset/bbc_news_text_complexity_summarization.csv \
  --model_name distilbert-base-uncased \
  --max_len 256 --batch_size 16 --lr 5e-5 \
  --weight_decay 0.01 --warmup_pct 0.10 \
  --epochs 5 --patience 2 --seed 42
```

**Outputs (per run)**

```
artifacts/classifier/<run_id>/best/   # model + tokenizer + metrics.json + label_map.json
reports/classification_report.txt     # appended per run
plots/confusion_matrix_<run_id>.png   # confusion matrix
```

---

## Use the model

### CLI

```bash
# Top-1 prediction from the most recent run
newsclf predict --text "Parliament approves the annual budget"

# Pick the best checkpoint across runs by a metric (e.g., macro-F1)
newsclf predict --text "Chipmaker unveils 3nm processor" --select best --metric test_macro_f1

# Show alternatives (top-3)
newsclf predict --text "Streaming platform renews hit fantasy series" --topk 3
```

### Python API

```python
from newsclf import NewsClassifier, predict

# Load once, predict many times (recommended for services)
clf = NewsClassifier(select="best", metric="test_macro_f1", max_len=256)
label, conf = clf.predict("Parliament approves the annual budget")
print(label, round(conf, 2))

# One-shot helper (loads each call)
print(predict("Star striker scores twice in cup final"))
```

---

## Test

Fast, deterministic end-to-end tests (tiny synthetic dataset):

```bash
pytest -q
# or verbose:
pytest -vv
```

Covers:

* Stratified data split & label mapping
* 1-epoch training smoke test
* Single-text inference from saved checkpoint
* CLI smoke test

---

## Project Flow (How it works)

1. **Load & split** the CSV (train/val/test = 70/15/15, stratified).
2. **Tokenize** with a HF tokenizer; batches use dynamic padding.
3. **Train** DistilBERT via a clean PyTorch loop (AdamW, linear warmup/decay, grad-clip).
4. **Early stop** on macro-F1; save **best** checkpoint for the run.
5. **Evaluate** on test set; write `metrics.json`, `classification_report.txt`, and the confusion matrix.
6. **Infer** via CLI or `NewsClassifier` (top-k confidences supported).

---

## Credits

BBC topic modeling exploration (BERTopic vs LDA) that inspired the dataset choice and comparison concept: Kaggle notebook by **Jacopo Ferretti**. ([Kaggle][1])

---

**Quickstart**

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -e .

# Train
newsclf train --data_csv dataset/bbc_news_text_complexity_summarization.csv --epochs 5

# Predict
newsclf predict --text "Parliament approves the annual budget"

# Test
pytest -q
```

[1]: https://www.kaggle.com/code/jacopoferretti/bbc-news-topic-modeling-with-bertopic-lda?utm_source=chatgpt.com "BBC News Topic Modeling with BERTopic & LDA"
