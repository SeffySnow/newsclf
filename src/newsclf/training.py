import json, time, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import load_and_split
from .datasets import NewsDataset
from .modeling import load_model_and_tokenizer, get_device
from .evaluate import eval_loop, save_confusion_matrix

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device()
    train_df, val_df, test_df, id2label, label2id = load_and_split(cfg.data_csv, cfg.seed)
    num_labels = len(id2label)

    model, tok = load_model_and_tokenizer(cfg.model_name, num_labels, id2label, label2id, device)
    collator = DataCollatorWithPadding(tokenizer=tok)

    train_dl = DataLoader(NewsDataset(train_df["text"], train_df["label_id"], tok, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_dl   = DataLoader(NewsDataset(val_df["text"],   val_df["label_id"],   tok, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
    test_dl  = DataLoader(NewsDataset(test_df["text"],  test_df["label_id"],  tok, cfg.max_len),
                          batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    decay_params, nodecay_params = [], []
    for n,p in model.named_parameters():
        (decay_params if not any(nd in n for nd in no_decay) else nodecay_params).append(p)
    optim = torch.optim.AdamW(
        [{"params":decay_params,"weight_decay":cfg.weight_decay},
         {"params":nodecay_params,"weight_decay":0.0}], lr=cfg.lr
    )
    total_steps = len(train_dl)*cfg.epochs; warmup_steps = int(total_steps*cfg.warmup_pct)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    art_dir = Path(cfg.artifacts_dir) / run_id
    best_dir = art_dir / "best"; best_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0; bad_epochs = 0; history=[]
    for epoch in range(1, cfg.epochs+1):
        model.train(); seen=0; loss_sum=0.0
        for batch in train_dl:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch); loss = out.loss
            loss.backward(); clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step(); sched.step(); optim.zero_grad()
            bs = batch["input_ids"].size(0); loss_sum += loss.item()*bs; seen += bs
        tr_loss = loss_sum/max(1,seen)
        val = eval_loop(model, val_dl, device)
        history.append({"epoch":epoch, "train_loss":tr_loss, **{k:v for k,v in val.items() if k not in ("y_true","y_pred")}})
        print(f"[{epoch}] train_loss={tr_loss:.4f} val_loss={val['loss']:.4f} macro_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]; bad_epochs = 0
            model.save_pretrained(best_dir); tok.save_pretrained(best_dir)
            (best_dir / "label_map.json").write_text(json.dumps(id2label, indent=2))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping after {cfg.patience} epoch(s) without improvement.")
                break

    # test on best
        # -------------------
 
    best_model = type(model).from_pretrained(best_dir).to(device)
    test = eval_loop(best_model, test_dl, device)

    metrics = {
        "run_id": run_id,
        "best_checkpoint": str(best_dir),
        "config": {
            "model_name": cfg.model_name, "max_len": cfg.max_len, "batch_size": cfg.batch_size,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay, "warmup_pct": cfg.warmup_pct,
            "epochs": cfg.epochs, "patience": cfg.patience, "seed": cfg.seed,
            "data_csv": cfg.data_csv,
        },
        "val_history": history,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_micro_f1": test["micro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "test_macro_precision": test["macro_precision"],
        "test_macro_recall": test["macro_recall"],
        "label_map": id2label,
    }

    # persist metrics in best/
    (best_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # plots dir
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # save confusion matrix with unique name; keep a "latest" copy too
    # NOTE: save_confusion_matrix returns a path; we then rename/copy to include run_id if needed.
    cm_tmp = save_confusion_matrix(
        plots_dir, test["y_true"], test["y_pred"], [id2label[i] for i in range(num_labels)]
    )
    cm_run = plots_dir / f"confusion_matrix_{run_id}.png"
    try:
        # if function already returned the desired name, this is a no-op copy
        Path(cm_tmp).replace(cm_run) if Path(cm_tmp) != cm_run else None
    except Exception:
        # fallback: just ensure the file exists under the run-specific name
        if Path(cm_tmp).exists() and not cm_run.exists():
            cm_run.write_bytes(Path(cm_tmp).read_bytes())
    # also keep a "latest" copy for convenience
    (plots_dir / "confusion_matrix.png").write_bytes(cm_run.read_bytes())

    # -------------------
    # reports dir
    # -------------------
    from sklearn.metrics import classification_report
    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # write unique, timestamped report + keep a latest copy
    report_txt = classification_report(
        test["y_true"], test["y_pred"],
        target_names=[id2label[i] for i in range(num_labels)],
        zero_division=0,  # avoid undefined metric warnings
    )
    report_run_path = reports_dir / f"classification_report_{run_id}.txt"
    report_run_path.write_text(report_txt)
    # also write/overwrite a "latest" pointer file
    (reports_dir / "classification_report.txt").write_text(report_txt)

    # store a copy of metrics in reports with the run id
    (reports_dir / f"metrics_{run_id}.json").write_text(json.dumps(metrics, indent=2))

    # append/initialize a CSV run summary
    import csv
    summary_csv = reports_dir / "runs_summary.csv"
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "run_id", "model_name", "max_len", "batch_size", "lr",
            "epochs", "patience", "test_accuracy", "test_macro_f1",
            "test_micro_f1", "test_weighted_f1", "data_csv"
        ])
        if write_header:
            w.writeheader()
        w.writerow({
            "run_id": run_id,
            "model_name": cfg.model_name,
            "max_len": cfg.max_len,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"],
            "test_micro_f1": metrics["test_micro_f1"],
            "test_weighted_f1": metrics["test_weighted_f1"],
            "data_csv": cfg.data_csv,
        })

    print(f"[DONE] report  : {report_run_path}")
    print(f"[DONE] metrics : {best_dir / 'metrics.json'} AND {reports_dir / f'metrics_{run_id}.json'}")
    print(f"[DONE] cm      : {cm_run}")

    return {"run_id": run_id, "best_dir": str(best_dir), "cm": str(cm_run), **metrics}
