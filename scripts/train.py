# scripts/train.py
from newsclf.config import TrainConfig
from newsclf.training import train

if __name__ == "__main__":
    cfg = TrainConfig(
        data_csv="dataset/bbc_news_text_complexity_summarization.csv",
        model_name="distilbert-base-uncased",
        epochs=2, max_len=128, batch_size=16, patience=1,
    )
    out = train(cfg)
    print("Best checkpoint:", out["best_checkpoint"])
    print("Test macro-F1:", round(out["test_macro_f1"], 3))
