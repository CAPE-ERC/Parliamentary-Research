"""Topic layer, Stage 2: fine-tuned classifier on the human-reviewed labels.

Trains on the 50 Policy-type topics from topic_taxonomy_labeled.xlsx as
distinct classes, with everything else (Governance/Procedure/Noise types)
collapsed into a single "non_policy" catch-all class. The catch-all is
downsampled before training (it's naturally ~69% of the corpus) to control
training time and reduce class-imbalance bias.

Base model is the same paraphrase-multilingual-MiniLM-L12-v2 checkpoint used
for the Stage-1 embeddings, for consistency and because it's small enough to
fine-tune on CPU in a bounded (if still multi-hour) time.

Usage:
    python -m topic_layer.train_classifier
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

RANDOM_STATE = 42
NON_POLICY_LABEL = "non_policy"


class UtteranceDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_labeled_dataset(processed_dir: Path) -> pd.DataFrame:
    """Join Stage-1 topic assignments with utterance text and human labels."""
    topics = pd.read_parquet(processed_dir / "topics.parquet")
    utterances = pd.read_parquet(processed_dir / "utterances.parquet")
    labeled = pd.read_excel(
        processed_dir / "topic_taxonomy_labeled.xlsx", sheet_name="Human-Labelled Taxonomy"
    )

    df = topics.merge(utterances[["debate_id", "seq_index", "text"]], on=["debate_id", "seq_index"])
    topic_to_label = labeled.set_index("topic_id")[["label", "label_type"]]
    df = df.merge(topic_to_label, left_on="topic_id", right_index=True, how="left")
    df["target"] = df.apply(
        lambda r: r["label"] if r["label_type"] == "Policy" else NON_POLICY_LABEL, axis=1
    )
    return df


def downsample_non_policy(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    non_policy = df[df["target"] == NON_POLICY_LABEL]
    policy = df[df["target"] != NON_POLICY_LABEL]
    if len(non_policy) > cap:
        non_policy = non_policy.sample(cap, random_state=RANDOM_STATE)
    return pd.concat([policy, non_policy]).reset_index(drop=True)


def run(config_path: Path, processed_dir: Path, models_dir: Path) -> None:
    config = load_config(config_path)
    clf_cfg = config.get("topic_layer", {}).get("classifier", {})
    model_name = clf_cfg.get("base_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    non_policy_cap = clf_cfg.get("non_policy_cap", 5000)
    num_epochs = clf_cfg.get("num_epochs", 3)
    max_length = clf_cfg.get("max_length", 64)
    batch_size = clf_cfg.get("batch_size", 16)

    print("Building labeled dataset from Stage 1 topics + human review...")
    full_df = build_labeled_dataset(processed_dir)
    train_df = downsample_non_policy(full_df, non_policy_cap)
    print(f"Training set: {len(train_df)} examples ({train_df['target'].nunique()} classes)")

    label_encoder = LabelEncoder()
    train_df = train_df.copy()
    train_df["label_id"] = label_encoder.fit_transform(train_df["target"])

    train_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=train_df["label_id"],
    )
    train_split = train_df.loc[train_idx]
    val_split = train_df.loc[val_idx]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(
        list(train_split["text"]), truncation=True, padding=True, max_length=max_length
    )
    val_enc = tokenizer(list(val_split["text"]), truncation=True, padding=True, max_length=max_length)

    train_ds = UtteranceDataset(train_enc, train_split["label_id"].values)
    val_ds = UtteranceDataset(val_enc, val_split["label_id"].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_encoder.classes_)
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(models_dir / "topic_classifier_checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)

    print(f"Fine-tuning {model_name} for {num_epochs} epochs on {len(train_split)} examples...")
    start = time.time()
    trainer.train()
    print(f"Training took {(time.time() - start) / 60:.1f} min")

    val_preds = trainer.predict(val_ds)
    y_pred = np.argmax(val_preds.predictions, axis=1)
    report = classification_report(
        val_split["label_id"].values, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print(report)

    model_dir = models_dir / "topic_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    with open(model_dir / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False, indent=2)

    with open(processed_dir / "classifier_eval_report.md", "w", encoding="utf-8") as f:
        f.write("# Topic Classifier (Stage 2) - Validation Report\n\n")
        f.write(f"- Base model: {model_name}\n")
        f.write(f"- Training examples: {len(train_split)} (validation: {len(val_split)})\n")
        f.write(f"- Classes: {len(label_encoder.classes_)} (50 Policy + non_policy catch-all)\n")
        f.write(f"- non_policy downsampled to: {non_policy_cap}\n\n")
        f.write("```\n" + report + "\n```\n")

    print("\nRunning inference over the full utterance set...")
    run_full_inference(full_df, model, tokenizer, label_encoder, max_length, batch_size, processed_dir)


def run_full_inference(
    full_df: pd.DataFrame,
    model,
    tokenizer,
    label_encoder: LabelEncoder,
    max_length: int,
    batch_size: int,
    processed_dir: Path,
) -> None:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = full_df["text"].tolist()
    all_preds, all_conf = [], []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size * 4):
            batch = texts[i : i + batch_size * 4]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            all_preds.extend(pred.cpu().tolist())
            all_conf.extend(conf.cpu().tolist())
            if i % (batch_size * 4 * 50) == 0:
                print(f"  inference {i}/{len(texts)}")

    out = full_df[["debate_id", "seq_index"]].copy()
    out["predicted_label"] = label_encoder.inverse_transform(all_preds)
    out["predicted_confidence"] = all_conf
    out.to_parquet(processed_dir / "utterance_policy_labels.parquet", index=False)

    print("\nPredicted label distribution:")
    print(out["predicted_label"].value_counts().head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    run(Path(args.config), Path(args.processed_dir), Path(args.models_dir))
