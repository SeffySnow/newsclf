from torch.utils.data import Dataset

class NewsDataset(Dataset):
    """Tokenizes on-the-fly; collator handles padding/tensorization."""
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts); self.labels = list(labels)
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = int(self.labels[i])
        return enc
