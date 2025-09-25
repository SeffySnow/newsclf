from dataclasses import dataclass

@dataclass
class TrainConfig:
    # backbone
    model_name: str = "distilbert-base-uncased"   # or "bert-base-uncased" if that’s what you had
    # sequence + batching
    max_len: int = 256
    batch_size: int = 16
    # optimization
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_pct: float = 0.10       
    grad_clip: float = 1.0
    # schedule / stopping
    epochs: int = 5
    patience: int = 2              
    # reproducibility
    seed: int = 42

    # io
    data_csv: str = "dataset/bbc_news_text_complexity_summarization.csv"
    artifacts_dir: str = "artifacts/classifier"
    reports_dir: str = "reports"
    plots_dir: str = "plots"
