#!/usr/bin/env python3
# scripts/mocking.py
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_latest_best(artifacts_root: str = "artifacts/classifier") -> Path:
    root = Path(artifacts_root)
    if not root.exists():
        raise FileNotFoundError(f"No artifacts directory found at: {root.resolve()}")
    # pick most recent run's `best/`
    bests = sorted(root.glob("*/best"), key=lambda p: p.stat().st_mtime)
    if not bests:
        raise FileNotFoundError(f"No best checkpoints found under {root.resolve()}")
    return bests[-1]


def load_id2label_from_config(model) -> dict[int, str]:
    # Robust to id2label being dict with string keys or a list
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        # coerce keys back to int if needed
        if not any(isinstance(k, int) for k in id2label.keys()):
            id2label = {int(k): v for k, v in id2label.items()}
        return id2label
    # list-like
    return {i: lab for i, lab in enumerate(id2label)}


def predict_label(checkpoint: Path, text: str, max_len: int = 256) -> tuple[str, float]:
    device = get_device()
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_id = int(np.argmax(probs))
    id2label = load_id2label_from_config(model)
    pred_label = id2label.get(pred_id, str(pred_id))
    return pred_label, float(probs[pred_id])


def main():
    ap = argparse.ArgumentParser(
        description="Predict the main topic of a news text using the latest trained checkpoint."
    )
    ap.add_argument("--text", type=str, help="Text to classify. If omitted, reads from stdin.")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a checkpoint dir (…/best). Defaults to latest under artifacts/classifier/")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    if args.text:
        text = args.text
    else:
        # read full stdin
        text = sys.stdin.read().strip()
        if not text:
            ap.error("No text provided. Use --text 'your text' or pipe input via stdin.")

    ckpt = Path(args.checkpoint) if args.checkpoint else find_latest_best()
    label, conf = predict_label(ckpt, text, max_len=args.max_len)

    print(f'this text main topic is "{label}" (confidence={conf:.2f})')


if __name__ == "__main__":
    main()
