"""Topic layer, Stage 2 (v2): two-stage classifier - Policy/non_policy gate,
then a 50-way domain classifier only on utterances the gate predicts Policy.

Rationale: the single 51-class baseline (train_classifier.py) likely spends
most of its errors on the non_policy catch-all swallowing/confusing edge
cases, not on telling the 50 policy domains apart. Splitting into a binary
gate + a domain classifier trained only on Policy examples tests that
hypothesis directly, without needing a bigger base model.

Fair-comparison design: the ORIGINAL baseline's reported 79% accuracy was
measured on a validation set drawn from its downsampled training pool (only
500 of 5472 examples were non_policy, ~9% - nothing like the true ~69%
non_policy prevalence in the corpus). That's not a representative number.
This script instead holds out one fixed validation set from the FULL,
naturally-distributed corpus (untouched by any downsampling) and evaluates
BOTH the baseline model and the new two-stage pipeline on it, so the
comparison is apples-to-apples against real-world class prevalence.

Usage:
    python -m topic_layer.train_two_stage
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from topic_layer.train_classifier import (
    NON_POLICY_LABEL,
    RANDOM_STATE,
    UtteranceDataset,
    build_labeled_dataset,
)


def fine_tune_classifier(
    train_df: pd.DataFrame,
    model_name: str,
    num_epochs: int,
    max_length: int,
    batch_size: int,
    output_dir: Path,
    target_col: str = "label_id",
) -> tuple:
    """Shared fine-tuning routine for both the binary gate and the domain
    classifier - same Trainer setup as train_classifier.py, parameterized."""
    train_idx, val_idx = train_test_split(
        train_df.index, test_size=0.1, random_state=RANDOM_STATE, stratify=train_df[target_col]
    )
    train_split, val_split = train_df.loc[train_idx], train_df.loc[val_idx]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(list(train_split["text"]), truncation=True, padding=True, max_length=max_length)
    val_enc = tokenizer(list(val_split["text"]), truncation=True, padding=True, max_length=max_length)

    train_ds = UtteranceDataset(train_enc, train_split[target_col].values)
    val_ds = UtteranceDataset(val_enc, val_split[target_col].values)

    n_classes = train_df[target_col].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
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

    print(f"Fine-tuning ({n_classes} classes) on {len(train_split)} examples...")
    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    print(f"Took {minutes:.1f} min")

    return model, tokenizer, minutes


def predict_batch(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> tuple:
    """Batched inference. Returns (pred_class_ids, confidences, full_probs)."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_conf, all_probs = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            all_preds.extend(pred.cpu().tolist())
            all_conf.extend(conf.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_conf), np.concatenate(all_probs)


def run(config_path: Path, processed_dir: Path, models_dir: Path) -> None:
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    clf_cfg = config.get("topic_layer", {}).get("classifier", {})
    two_stage_cfg = config.get("topic_layer", {}).get("two_stage", {})
    model_name = clf_cfg.get("base_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_length = clf_cfg.get("max_length", 64)
    batch_size = clf_cfg.get("batch_size", 16)
    gate_non_policy_cap = two_stage_cfg.get("gate_non_policy_cap", 45000)
    gate_epochs = two_stage_cfg.get("gate_epochs", 2)
    domain_epochs = two_stage_cfg.get("domain_epochs", 3)

    print("Building labeled dataset...")
    full_df = build_labeled_dataset(processed_dir)
    full_df = full_df.reset_index(drop=True)

    # ONE fixed, naturally-distributed held-out set for fair comparison -
    # not touched by any downsampling done for either model's training.
    natural_train_idx, natural_val_idx = train_test_split(
        full_df.index, test_size=0.1, random_state=RANDOM_STATE, stratify=full_df["target"]
    )
    natural_train = full_df.loc[natural_train_idx].reset_index(drop=True)
    natural_val = full_df.loc[natural_val_idx].reset_index(drop=True)
    print(f"Natural split: {len(natural_train)} train / {len(natural_val)} held-out "
          f"({(natural_val['target'] == NON_POLICY_LABEL).mean() * 100:.1f}% non_policy in held-out set)")

    # ---- Stage A: binary Policy vs non_policy gate ----
    gate_df = natural_train.copy()
    gate_df["is_policy"] = (gate_df["target"] != NON_POLICY_LABEL).astype(int)
    non_policy_rows = gate_df[gate_df["is_policy"] == 0]
    policy_rows = gate_df[gate_df["is_policy"] == 1]
    if len(non_policy_rows) > gate_non_policy_cap:
        non_policy_rows = non_policy_rows.sample(gate_non_policy_cap, random_state=RANDOM_STATE)
    gate_train_df = pd.concat([policy_rows, non_policy_rows]).reset_index(drop=True)
    gate_train_df = gate_train_df.rename(columns={"is_policy": "label_id"})
    print(f"\n=== Stage A: Policy/non_policy gate ({len(gate_train_df)} examples) ===")

    gate_model, gate_tokenizer, gate_minutes = fine_tune_classifier(
        gate_train_df, model_name, gate_epochs, max_length, batch_size, models_dir / "gate_tmp"
    )
    gate_dir = models_dir / "topic_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_model.save_pretrained(str(gate_dir))
    gate_tokenizer.save_pretrained(str(gate_dir))

    # ---- Stage B: 50-way domain classifier, Policy examples only ----
    domain_df = natural_train[natural_train["target"] != NON_POLICY_LABEL].copy()
    domain_le = LabelEncoder()
    domain_df["label_id"] = domain_le.fit_transform(domain_df["target"])
    print(f"\n=== Stage B: 50-way domain classifier ({len(domain_df)} Policy examples) ===")

    domain_model, domain_tokenizer, domain_minutes = fine_tune_classifier(
        domain_df, model_name, domain_epochs, max_length, batch_size, models_dir / "domain_tmp"
    )
    domain_dir = models_dir / "topic_domain_classifier"
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_model.save_pretrained(str(domain_dir))
    domain_tokenizer.save_pretrained(str(domain_dir))
    with open(domain_dir / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(list(domain_le.classes_), f, ensure_ascii=False, indent=2)

    # ---- Fair comparison on the shared natural-distribution held-out set ----
    print("\n=== Evaluating two-stage pipeline on natural-distribution held-out set ===")
    texts = natural_val["text"].tolist()
    gate_preds, gate_conf, _ = predict_batch(gate_model, gate_tokenizer, texts, max_length, batch_size)

    two_stage_pred_labels = np.array([NON_POLICY_LABEL] * len(natural_val), dtype=object)
    policy_gate_idx = np.where(gate_preds == 1)[0]
    if len(policy_gate_idx) > 0:
        policy_texts = [texts[i] for i in policy_gate_idx]
        domain_preds, _, _ = predict_batch(domain_model, domain_tokenizer, policy_texts, max_length, batch_size)
        two_stage_pred_labels[policy_gate_idx] = domain_le.inverse_transform(domain_preds)

    y_true = natural_val["target"].values
    two_stage_acc = accuracy_score(y_true, two_stage_pred_labels)
    two_stage_f1 = f1_score(y_true, two_stage_pred_labels, average="macro", zero_division=0)
    two_stage_report = classification_report(y_true, two_stage_pred_labels, zero_division=0)

    baseline_dir = models_dir / "topic_classifier"
    baseline_report_text = "Baseline model not found - skipping comparison."
    baseline_acc = baseline_f1 = None
    if baseline_dir.exists():
        print("Evaluating original single-stage baseline on the SAME held-out set...")
        baseline_model = AutoModelForSequenceClassification.from_pretrained(str(baseline_dir))
        baseline_tokenizer = AutoTokenizer.from_pretrained(str(baseline_dir))
        with open(baseline_dir / "label_encoder.json", encoding="utf-8") as f:
            baseline_classes = json.load(f)
        baseline_preds, _, _ = predict_batch(baseline_model, baseline_tokenizer, texts, max_length, batch_size)
        baseline_pred_labels = np.array(baseline_classes)[baseline_preds]
        baseline_acc = accuracy_score(y_true, baseline_pred_labels)
        baseline_f1 = f1_score(y_true, baseline_pred_labels, average="macro", zero_division=0)
        baseline_report_text = classification_report(y_true, baseline_pred_labels, zero_division=0)

    with open(processed_dir / "two_stage_vs_baseline_report.md", "w", encoding="utf-8") as f:
        f.write("# Two-Stage vs Single-Stage Classifier - Fair Comparison\n\n")
        f.write(
            "Both models evaluated on the SAME held-out set, drawn from the full, "
            "naturally-distributed corpus (not the downsampled training pool), so this "
            f"is apples-to-apples. Held-out set: {len(natural_val)} examples, "
            f"{(natural_val['target'] == NON_POLICY_LABEL).mean() * 100:.1f}% non_policy "
            "(true corpus prevalence).\n\n"
        )
        f.write("## Headline numbers\n\n")
        f.write("| Model | Accuracy | Macro F1 |\n|---|---|---|\n")
        if baseline_acc is not None:
            f.write(f"| Single-stage baseline (51-class) | {baseline_acc:.3f} | {baseline_f1:.3f} |\n")
        f.write(f"| Two-stage (gate + domain classifier) | {two_stage_acc:.3f} | {two_stage_f1:.3f} |\n\n")
        f.write(f"Training time: gate {gate_minutes:.1f} min, domain classifier {domain_minutes:.1f} min\n\n")
        f.write("## Two-stage pipeline - full classification report\n\n```\n" + two_stage_report + "\n```\n\n")
        f.write("## Single-stage baseline - full classification report (same held-out set)\n\n```\n"
                 + baseline_report_text + "\n```\n")

    print(f"\nBaseline:  acc={baseline_acc}, macro_f1={baseline_f1}")
    print(f"Two-stage: acc={two_stage_acc:.3f}, macro_f1={two_stage_f1:.3f}")

    # ---- Full-corpus inference with the two-stage pipeline ----
    print("\nRunning two-stage inference over the full utterance set...")
    all_texts = full_df["text"].tolist()
    gate_preds_full, gate_conf_full, _ = predict_batch(gate_model, gate_tokenizer, all_texts, max_length, batch_size)

    final_labels = np.array([NON_POLICY_LABEL] * len(full_df), dtype=object)
    final_conf = gate_conf_full.copy()
    policy_idx_full = np.where(gate_preds_full == 1)[0]
    if len(policy_idx_full) > 0:
        policy_texts_full = [all_texts[i] for i in policy_idx_full]
        domain_preds_full, domain_conf_full, _ = predict_batch(
            domain_model, domain_tokenizer, policy_texts_full, max_length, batch_size
        )
        final_labels[policy_idx_full] = domain_le.inverse_transform(domain_preds_full)
        final_conf[policy_idx_full] = domain_conf_full

    out = full_df[["debate_id", "seq_index"]].copy()
    out["predicted_label"] = final_labels
    out["predicted_confidence"] = final_conf
    out.to_parquet(processed_dir / "utterance_policy_labels_two_stage.parquet", index=False)
    print("\nPredicted label distribution (two-stage):")
    print(out["predicted_label"].value_counts().head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    run(Path(args.config), Path(args.processed_dir), Path(args.models_dir))
