import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model_and_tokenizer(model_name: str, num_labels: int, id2label: dict, label2id: dict, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)
    return model, tok
