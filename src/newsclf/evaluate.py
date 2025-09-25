import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

def eval_loop(model, dataloader, device):
    model.eval()
    tot_loss = 0.0; preds_all=[]; labels_all=[]
    import torch
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            tot_loss += out.loss.item() * batch["input_ids"].size(0)
            preds_all.append(out.logits.argmax(-1).cpu().numpy())
            labels_all.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(preds_all); y_true = np.concatenate(labels_all)
    return {
        "loss": tot_loss/len(dataloader.dataset),
        "accuracy": accuracy_score(y_true,y_pred),
        "macro_f1": f1_score(y_true,y_pred,average="macro"),
        "micro_f1": f1_score(y_true,y_pred,average="micro"),
        "weighted_f1": f1_score(y_true,y_pred,average="weighted"),
        "macro_precision": precision_score(y_true,y_pred,average="macro",zero_division=0),
        "macro_recall": recall_score(y_true,y_pred,average="macro",zero_division=0),
        "y_true": y_true, "y_pred": y_pred
    }

def save_confusion_matrix(cm_dir: Path, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    import numpy as np
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right"); plt.yticks(ticks, labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    out = cm_dir / "confusion_matrix.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out
