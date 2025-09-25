from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_split(csv_path: str, seed: int = 42):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")

    df = pd.read_csv(p)[["text", "labels"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["labels"])
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label_id"]
    )
    id2label = {int(i): lab for i, lab in enumerate(le.classes_)}
    label2id = {v: k for k, v in id2label.items()}
    return train_df, val_df, test_df, id2label, label2id
