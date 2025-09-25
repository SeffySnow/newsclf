# tests/test_all.py
"""
End-to-end tests for the news classifier in one file.

Covers:
- Data load & stratified split
- Training smoke test (1 epoch, small MAX_LEN)
- Single-text inference on saved checkpoint (reads labels from model.config)
- CLI entrypoint smoke test (skipped if CLI module absent)
"""

import sys, json, random, os
from pathlib import Path
import numpy as np
import pytest
import torch

# Make 'newsclf' importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

# Core imports from your module
from newsclf.config import TrainConfig
from newsclf.data import load_and_split
from newsclf.training import train

# Optional CLI import (skip tests if not present)
try:
    from newsclf import main as newscli
    HAS_CLI = True
except Exception:
    HAS_CLI = False

# ---------------- Fixtures ----------------
@pytest.fixture(autouse=True)
def seeded():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def tmp_dirs(tmp_path: Path):
    arts = tmp_path / "artifacts"; reps = tmp_path / "reports"; plots = tmp_path / "plots"
    arts.mkdir(parents=True, exist_ok=True)
    reps.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return {"artifacts": str(arts), "reports": str(reps), "plots": str(plots)}

@pytest.fixture
def tiny_csv(tmp_path: Path) -> str:
    rows = []
    def add(label, texts):
        for t in texts: rows.append((t, label))

    add("sport", [
        "Home team wins the football match by two goals",
        "Tennis champion secures title at Wimbledon",
        "Coach praises defense after basketball victory",
        "Striker scores hat-trick in derby",
        "Fans celebrate dramatic rugby comeback",
        "Cricket captain leads series win",
        "Olympic runners break national record",
        "Boxer wins by unanimous decision",
        "Marathon event attracts thousands",
        "Club signs star midfielder",
    ])
    add("business", [
        "FTSE rises as bank stocks rally",
        "Tech firm announces quarterly earnings beat",
        "Shareholders approve merger deal",
        "Inflation report boosts currency markets",
        "CEO outlines growth strategy",
        "Retail sales surge in holiday season",
        "Oil prices fall amid demand concerns",
        "Startups raise record venture funding",
        "Central bank maintains interest rates",
        "Company launches cost-cutting plan",
    ])
    add("politics", [
        "Parliament debates new immigration bill",
        "Prime minister addresses the press",
        "Opposition calls for vote of no confidence",
        "Election polls show tight race",
        "Lawmakers propose tax reform",
        "Government faces ethics inquiry",
        "Policy speech highlights climate agenda",
        "Senate committee schedules hearings",
        "Diplomatic talks resume after pause",
        "Court rules on campaign finance case",
    ])
    add("entertainment", [
        "Actor announces new film project",
        "Festival celebrates independent cinema",
        "Singer releases chart-topping single",
        "Critics praise theatre performance",
        "Award show honors best television series",
        "Director shares behind-the-scenes footage",
        "Fans react to surprise album drop",
        "Comedian sells out tour dates",
        "Documentary explores pop culture icons",
        "Streaming service unveils original drama",
    ])
    add("tech", [
        "Startup unveils AI-powered chipset",
        "Cybersecurity update patches major flaw",
        "Smartphone manufacturer teases new model",
        "Researchers publish breakthrough in quantum computing",
        "Developers adopt open-source framework",
        "Cloud provider expands data centers",
        "Robotics firm demonstrates warehouse automation",
        "Semiconductor demand rebounds in Q3",
        "VR headset adds hand-tracking feature",
        "Autonomous vehicle software receives update",
    ])

    import pandas as pd
    df = pd.DataFrame(rows, columns=["text", "labels"])
    out = tmp_path / "tiny_bbc.csv"
    df.to_csv(out, index=False)
    return str(out)

# ---------------- Tests ----------------
def test_load_and_split_shapes(tiny_csv):
    tr, va, te, id2label, label2id = load_and_split(tiny_csv, seed=42)
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    for col in ("text", "labels", "label_id"):
        assert col in tr.columns
    assert set(id2label.values()) == set(label2id.keys())
    assert tr["labels"].nunique() == len(id2label)
    assert va["labels"].nunique() == len(id2label)
    assert te["labels"].nunique() == len(id2label)

def test_train_smoke(tmp_dirs, tiny_csv):
    cfg = TrainConfig(
        data_csv=tiny_csv, model_name="distilbert-base-uncased", epochs=1, max_len=64, batch_size=8,
        artifacts_dir=tmp_dirs["artifacts"], reports_dir=tmp_dirs["reports"], plots_dir=tmp_dirs["plots"],
        patience=1,
    )
    out = train(cfg)
    for k in ("best_checkpoint", "test_accuracy", "test_macro_f1", "label_map"):
        assert k in out
    best = Path(out["best_checkpoint"])
    assert best.exists()
    assert (best / "metrics.json").exists()
    assert Path(tmp_dirs["reports"], "classification_report.txt").exists()
    assert 0.0 <= out["test_accuracy"] <= 1.0
    assert 0.0 <= out["test_macro_f1"] <= 1.0

def test_single_inference_after_train(tmp_dirs, tiny_csv):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Train a tiny model fast
    cfg = TrainConfig(
        data_csv=tiny_csv, model_name="distilbert-base-uncased", epochs=1, max_len=64, batch_size=8,
        artifacts_dir=tmp_dirs["artifacts"], reports_dir=tmp_dirs["reports"], plots_dir=tmp_dirs["plots"],
        patience=1,
    )
    out = train(cfg)
    best = Path(out["best_checkpoint"])

    # Load trained checkpoint
    tok = AutoTokenizer.from_pretrained(best)
    model = AutoModelForSequenceClassification.from_pretrained(best)
    model.eval()

    # Predict
    enc = tok("The club confirmed the striker transfer after the football match.",
              return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**enc).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())

    # Get label name from model.config.id2label (robust to str/int keys)
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        if not any(isinstance(k, int) for k in id2label.keys()):
            id2label = {int(k): v for k, v in id2label.items()}
        pred_label = id2label[pred_id]
    else:
        pred_label = id2label[pred_id]

    assert pred_label in {"sport","business","politics","entertainment","tech"}

@pytest.mark.skipif(not HAS_CLI, reason="CLI module not present")
def test_cli_smoke(tmp_dirs, tiny_csv, monkeypatch):
    # Use the new subcommand-style CLI: newsclf train ...
    argv = [
        "newsclf", "train",
        "--data_csv", tiny_csv,
        "--epochs", "1",
        "--max_len", "64",
        "--batch_size", "8",
        "--artifacts_dir", tmp_dirs["artifacts"],
        "--reports_dir", tmp_dirs["reports"],
        "--plots_dir", tmp_dirs["plots"],
    ]
    monkeypatch.setattr(sys, "argv", argv)
    newscli.cli()  # should not raise
