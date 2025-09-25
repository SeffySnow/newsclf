# src/newsclf/inference.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_id2label(cfg) -> Dict[int, str]:
    m = cfg.id2label
    if isinstance(m, dict):
        return {int(k): v for k, v in m.items()}
    return {i: lab for i, lab in enumerate(m)}


def _find_checkpoint(artifacts_root: str = "artifacts/classifier",
                     select: str = "latest",
                     metric: str = "test_macro_f1") -> Path:
    """
    select='latest' -> newest run's best/
    select='best'   -> highest `metric` across runs (reads best/metrics.json)
    """
    root = Path(artifacts_root)
    best_dirs = sorted(root.glob("*/best"))
    if not root.exists() or not best_dirs:
        raise FileNotFoundError(f"No best checkpoints under {root.resolve()}")

    if select == "latest":
        return sorted(best_dirs, key=lambda p: p.stat().st_mtime)[-1]

    # select best overall by metric
    import json
    best_score, best_path = None, None
    for b in best_dirs:
        m = b / "metrics.json"
        if not m.exists():
            continue
        try:
            data = json.loads(m.read_text())
            score = data.get(metric, None)
            if score is None:
                continue
            if (best_score is None) or (score > best_score) or (
                score == best_score and b.stat().st_mtime > best_path.stat().st_mtime
            ):
                best_score, best_path = score, b
        except Exception:
            pass
    return best_path or sorted(best_dirs, key=lambda p: p.stat().st_mtime)[-1]


class NewsClassifier:
    """Load once, predict many times."""
    def __init__(self,
                 checkpoint: Optional[str | Path] = None,
                 max_len: int = 256,
                 select: str = "latest",
                 metric: str = "test_macro_f1",
                 artifacts_root: str = "artifacts/classifier"):
        self.ckpt = Path(checkpoint) if checkpoint else _find_checkpoint(
            artifacts_root=artifacts_root, select=select, metric=metric
        )
        self.max_len = max_len
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()
        self.id2label = _normalize_id2label(self.model.config)

    def predict(self, text: str, topk: int = 1) -> Tuple[str, float] | List[Tuple[str, float]]:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0].cpu().numpy()
        order = np.argsort(-probs)[:topk]
        if topk == 1:
            i = int(order[0]); return self.id2label[i], float(probs[i])
        return [(self.id2label[int(i)], float(probs[int(i)])) for i in order]


def predict(text: str,
            checkpoint: Optional[str | Path] = None,
            max_len: int = 256,
            topk: int = 1,
            select: str = "latest",
            metric: str = "test_macro_f1",
            artifacts_root: str = "artifacts/classifier"):
    """One-shot helper (loads model each call). Prefer NewsClassifier for repeated calls."""
    clf = NewsClassifier(checkpoint=checkpoint, max_len=max_len, select=select,
                         metric=metric, artifacts_root=artifacts_root)
    return clf.predict(text, topk=topk)
