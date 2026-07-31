"""Trains the BiLSTM multi-label classifier on the rule-tagger's silver
labels, evaluates held-out agreement, runs full-corpus inference, and exports
rule/LSTM disagreements for human review.

No human-annotated gold set exists for this layer (unlike the Topic layer) -
see procedural_layer_report.md for why. Held-out "accuracy" here measures
whether the LSTM learned to reproduce the RULES' patterns, which confirms
training worked, but is not an independent real-world accuracy figure. The
disagreement sample is the more informative artifact: cases where the LSTM
generalized beyond exact keyword/structural matches, or made a mistake.

Usage:
    python -m procedural_layer.train_lstm
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .lstm_model import ProceduralLSTM
from .pipeline import TAG_COLUMNS
from .vocab import build_vocab, encode

RANDOM_STATE = 42


class TaggedUtteranceDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, labels: np.ndarray):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.labels[idx]


def load_silver_dataset(processed_dir: Path) -> pd.DataFrame:
    utterances = pd.read_parquet(processed_dir / "utterances.parquet")
    tags = pd.read_parquet(processed_dir / "procedural_tags_rules.parquet")
    return utterances.merge(tags, on=["debate_id", "seq_index"])


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    return np.array([encode(t, vocab, max_len) for t in texts], dtype=np.int64)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    num_epochs: int,
    lr: float,
) -> None:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for token_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for token_ids, labels in val_loader:
                logits = model(token_ids)
                val_loss += criterion(logits, labels).item() * len(labels)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for (token_ids,) in loader:
            probs = torch.sigmoid(model(token_ids))
            all_probs.append(probs.numpy())
    return np.concatenate(all_probs)


class InferenceDataset(Dataset):
    def __init__(self, token_ids: np.ndarray):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        return (self.token_ids[idx],)


def run(config_path: Path, processed_dir: Path, models_dir: Path) -> None:
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    lstm_cfg = config.get("procedural_layer", {}).get("lstm", {})
    embedding_dim = lstm_cfg.get("embedding_dim", 100)
    hidden_dim = lstm_cfg.get("hidden_dim", 128)
    max_vocab = lstm_cfg.get("max_vocab_size", 20_000)
    max_len = lstm_cfg.get("max_tokens", 50)
    num_epochs = lstm_cfg.get("num_epochs", 5)
    batch_size = lstm_cfg.get("batch_size", 64)
    lr = lstm_cfg.get("learning_rate", 1e-3)
    max_pos_weight = lstm_cfg.get("max_pos_weight", 20.0)

    print("Loading silver-labeled dataset...")
    df = load_silver_dataset(processed_dir).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df[TAG_COLUMNS].values.astype(np.float32)

    print("Building vocabulary...")
    vocab = build_vocab(texts, max_size=max_vocab)
    print(f"Vocab size: {len(vocab)}")

    print("Encoding texts...")
    token_ids = encode_texts(texts, vocab, max_len)

    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.1, random_state=RANDOM_STATE)
    train_ds = TaggedUtteranceDataset(token_ids[train_idx], labels[train_idx])
    val_ds = TaggedUtteranceDataset(token_ids[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    n_pos = labels[train_idx].sum(axis=0)
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor(np.clip(n_neg / np.maximum(n_pos, 1), 1.0, max_pos_weight), dtype=torch.float32)
    print(f"pos_weight per tag: {dict(zip(TAG_COLUMNS, pos_weight.tolist()))}")

    torch.manual_seed(RANDOM_STATE)
    model = ProceduralLSTM(len(vocab), len(TAG_COLUMNS), embedding_dim, hidden_dim)

    print(f"\nTraining for {num_epochs} epochs on {len(train_idx)} examples...")
    start = time.time()
    train_model(model, train_loader, val_loader, pos_weight, num_epochs, lr)
    print(f"Training took {(time.time() - start) / 60:.1f} min")

    # ---- Evaluate held-out agreement with the rule tagger ----
    val_probs = predict(model, DataLoader(InferenceDataset(token_ids[val_idx]), batch_size=batch_size * 2))
    val_preds = (val_probs >= 0.5).astype(int)
    val_true = labels[val_idx].astype(int)

    report_lines = ["# Procedural Layer - LSTM Training Report", ""]
    report_lines.append(f"- Vocabulary size: {len(vocab)}")
    report_lines.append(f"- Training examples: {len(train_idx)} (held-out: {len(val_idx)})")
    report_lines.append(
        "- **No human-annotated gold set** - trained on the rule-tagger's silver labels. "
        "The metrics below measure agreement with those rules (confirms training worked), "
        "not independent real-world accuracy."
    )
    report_lines.append("")
    report_lines.append("## Held-out agreement with rule-tagger labels")
    report_lines.append("")
    report_lines.append("| Tag | Precision | Recall | F1 |")
    report_lines.append("|---|---|---|---|")
    for i, col in enumerate(TAG_COLUMNS):
        p = precision_score(val_true[:, i], val_preds[:, i], zero_division=0)
        r = recall_score(val_true[:, i], val_preds[:, i], zero_division=0)
        f1 = f1_score(val_true[:, i], val_preds[:, i], zero_division=0)
        report_lines.append(f"| {col} | {p:.3f} | {r:.3f} | {f1:.3f} |")

    # ---- Full-corpus inference ----
    print("\nRunning full-corpus inference...")
    full_loader = DataLoader(InferenceDataset(token_ids), batch_size=batch_size * 4)
    full_probs = predict(model, full_loader)
    full_preds = (full_probs >= 0.5).astype(int)

    lstm_df = df[["debate_id", "seq_index"]].copy()
    for i, col in enumerate(TAG_COLUMNS):
        lstm_df[f"{col}_rule"] = df[col].astype(int)
        lstm_df[f"{col}_lstm"] = full_preds[:, i]
        lstm_df[f"{col}_combined"] = (df[col].astype(int) | full_preds[:, i]).astype(int)

    lstm_df.to_parquet(processed_dir / "procedural_tags_final.parquet", index=False)

    # ---- Disagreement sample for human review ----
    disagreements = []
    for i, col in enumerate(TAG_COLUMNS):
        mask = df[col].astype(int).values != full_preds[:, i]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(idx, size=min(20, len(idx)), replace=False)
        for j in sample_idx:
            disagreements.append(
                {
                    "tag": col,
                    "debate_id": df.iloc[j]["debate_id"],
                    "seq_index": df.iloc[j]["seq_index"],
                    "rule_says": int(df.iloc[j][col]),
                    "lstm_says": int(full_preds[j, i]),
                    "lstm_confidence": round(float(full_probs[j, i]), 3),
                    "text": df.iloc[j]["text"][:300],
                }
            )
    pd.DataFrame(disagreements).to_csv(processed_dir / "procedural_disagreements_sample.csv", index=False)

    report_lines.append("")
    report_lines.append("## Full-corpus tag counts (combined = rule OR lstm)")
    report_lines.append("")
    report_lines.append("| Tag | Rule count | LSTM count | Combined count |")
    report_lines.append("|---|---|---|---|")
    for col in TAG_COLUMNS:
        report_lines.append(
            f"| {col} | {lstm_df[f'{col}_rule'].sum()} | {lstm_df[f'{col}_lstm'].sum()} | "
            f"{lstm_df[f'{col}_combined'].sum()} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Disagreement sample (up to 20 per tag) written to "
        "procedural_disagreements_sample.csv for spot-checking."
    )
    (processed_dir / "procedural_layer_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "procedural_lstm.pt")
    with open(models_dir / "procedural_lstm_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print("\n".join(report_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    run(Path(args.config), Path(args.processed_dir), Path(args.models_dir))
