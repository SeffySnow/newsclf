# src/newsclf/main.py
from __future__ import annotations
import argparse
import sys
from .config import TrainConfig
from .training import train
from .inference import NewsClassifier

def _add_train_args(p: argparse.ArgumentParser):
    p.add_argument("--data_csv", required=True, help="Path to BBC CSV (text, labels).")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_pct", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts_dir", default="artifacts/classifier")
    p.add_argument("--reports_dir", default="reports")
    p.add_argument("--plots_dir", default="plots")

def _add_predict_args(p: argparse.ArgumentParser):
    p.add_argument("--text", help="Text to classify; if omitted, read from stdin.")
    p.add_argument("--checkpoint", help="Path to .../best; overrides selection logic.")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--select", choices=["latest", "best"], default="latest",
                   help="How to choose a checkpoint when --checkpoint not provided.")
    p.add_argument("--metric", default="test_macro_f1",
                   help="Metric to rank by when --select=best (must be in metrics.json).")
    p.add_argument("--artifacts_root", default="artifacts/classifier",
                   help="Root folder containing run subdirs.")

def cli(argv=None):
    ap = argparse.ArgumentParser("newsclf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train", help="Train a classifier on BBC data.")
    _add_train_args(p_train)
    p_pred = sub.add_parser("predict", help="Predict topic for a given text.")
    _add_predict_args(p_pred)
    args = ap.parse_args(argv)

    if args.cmd == "train":
        cfg = TrainConfig(
            data_csv=args.data_csv,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_pct=args.warmup_pct,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            reports_dir=args.reports_dir,
            plots_dir=args.plots_dir,
        )
        out = train(cfg)
        print(f"\n✅ trained. best={out['best_checkpoint']} macro_f1={out['test_macro_f1']:.3f}")
        return

    if args.cmd == "predict":
        text = args.text or sys.stdin.read().strip()
        if not text:
            ap.error("Provide --text or pipe text via stdin.")
        clf = NewsClassifier(
            checkpoint=args.checkpoint,
            max_len=args.max_len,
            select=args.select,
            metric=args.metric,
            artifacts_root=args.artifacts_root,
        )
        pred = clf.predict(text, topk=args.topk)
        if isinstance(pred, tuple):
            lab, p = pred
            print(f'this text main topic is "{lab}" (confidence={p:.2f})')
        else:
            lab, p = pred[0]
            print(f'this text main topic is "{lab}" (confidence={p:.2f})')
            if len(pred) > 1:
                alts = ", ".join([f"{l}={pp:.2f}" for l, pp in pred[1:]])
                print(f"alternatives: {alts}")

if __name__ == "__main__":
    cli()
