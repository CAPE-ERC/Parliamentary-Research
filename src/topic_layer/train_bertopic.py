"""Topic layer, Stage 1: unsupervised topic discovery with BERTopic.

Fits BERTopic over substantive dialogue utterances (stage directions and
very short utterances excluded) using a multilingual sentence-transformer
embedding, since the corpus code-switches between English, French, and
Kreol Morisien. Outputs a topic taxonomy for human review/labeling - Stage 2
(fine-tuning a classifier on the reviewed labels) follows once this is
validated.

Usage:
    python -m topic_layer.train_bertopic
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sentence_transformers import SentenceTransformer
from umap import UMAP

RANDOM_STATE = 42


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_corpus(utterances_path: Path, min_chars: int) -> pd.DataFrame:
    df = pd.read_parquet(utterances_path)
    df = df[~df["is_stage_direction"]]
    df = df[df["text"].str.len() >= min_chars]
    return df.reset_index(drop=True)


def compute_or_load_embeddings(
    texts: list[str], embedding_model_name: str, cache_path: Path
) -> np.ndarray:
    """Cache embeddings to disk - re-clustering with different min_topic_size
    / nr_topics is then ~2 min instead of ~30 min, since embedding is the
    expensive step at this corpus size (~160k utterances)."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"Loaded cached embeddings from {cache_path} ({cached.shape}).")
            return cached
        print("Cached embeddings size mismatch with corpus - recomputing.")

    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def fit_topics(
    texts: list[str],
    embeddings: np.ndarray,
    min_topic_size: int,
    nr_topics: int | str | None = None,
) -> BERTopic:
    # Fixed random_state: UMAP is stochastic by default, so without it,
    # re-running produces different topic assignments each time - a problem
    # once a specific run's taxonomy is being hand-labeled by the team.
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=RANDOM_STATE
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topic_model.fit(texts, embeddings=embeddings)
    return topic_model


def compute_coherence_per_topic(
    topic_model: BERTopic,
    texts: list[str],
    topic_ids: list[int],
    top_n: int = 10,
    sample_size: int = 50_000,
) -> dict[int, float]:
    """c_v coherence per topic (0-1, higher = more interpretable word co-occurrence).

    Built from a random subsample of the corpus for the co-occurrence
    statistics c_v relies on - full-corpus computation over ~160k documents
    is unnecessary for stable estimates and much slower.
    """
    rng = random.Random(RANDOM_STATE)
    sample = texts if len(texts) <= sample_size else rng.sample(texts, sample_size)
    tokenized = [t.lower().split() for t in sample]
    dictionary = Dictionary(tokenized)

    topics_words = [
        [w for w, _ in topic_model.get_topic(tid)[:top_n]] if tid != -1 else []
        for tid in topic_ids
    ]
    # CoherenceModel chokes on empty topic-word lists (the -1/outlier row).
    valid_idx = [i for i, words in enumerate(topics_words) if words]
    cm = CoherenceModel(
        topics=[topics_words[i] for i in valid_idx],
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    scores = cm.get_coherence_per_topic()

    result: dict[int, float] = {}
    for pos, i in enumerate(valid_idx):
        result[topic_ids[i]] = scores[pos]
    return result


def write_taxonomy(
    topic_model: BERTopic,
    df: pd.DataFrame,
    topics: list[int],
    coherence: dict[int, float],
    out_path: Path,
) -> None:
    info = topic_model.get_topic_info()
    rows = []
    for _, row in info.iterrows():
        topic_id = row["Topic"]
        top_words = ", ".join(w for w, _ in topic_model.get_topic(topic_id)[:10]) if topic_id != -1 else ""
        examples = df.loc[[i for i, t in enumerate(topics) if t == topic_id]]["text"].head(3).tolist()
        rows.append(
            {
                "topic_id": topic_id,
                "count": row["Count"],
                "coherence_c_v": round(coherence.get(topic_id, float("nan")), 4),
                "label": "",  # for the team to fill in with a human-readable policy-domain name
                "top_words": top_words,
                "example_1": examples[0] if len(examples) > 0 else "",
                "example_2": examples[1] if len(examples) > 1 else "",
                "example_3": examples[2] if len(examples) > 2 else "",
            }
        )
    pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(out_path, index=False)


def run(config_path: Path, processed_dir: Path, models_dir: Path) -> None:
    config = load_config(config_path)
    topic_cfg = config.get("topic_layer", {})

    df = load_corpus(
        processed_dir / "utterances.parquet",
        topic_cfg.get("min_utterance_chars_for_topics", 10),
    )
    print(f"Fitting BERTopic on {len(df)} utterances...")

    embeddings = compute_or_load_embeddings(
        df["text"].tolist(),
        topic_cfg.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
        models_dir / "embeddings.npy",
    )
    topic_model = fit_topics(
        df["text"].tolist(),
        embeddings,
        topic_cfg.get("min_topic_size", 15),
        topic_cfg.get("nr_topics"),
    )
    topics = topic_model.topics_
    n_outliers_before = sum(1 for t in topics if t == -1)

    # The framework's task is to label EVERY utterance, not leave ~half
    # unassigned - reassign outliers to their nearest topic by embedding
    # similarity (reuses the cached embeddings, no re-encoding needed).
    print(f"Reassigning {n_outliers_before} outliers to nearest topic...")
    texts = df["text"].tolist()
    topics = topic_model.reduce_outliers(texts, topics, strategy="embeddings", embeddings=embeddings)
    topic_model.update_topics(texts, topics=topics)

    df = df.copy()
    df["topic_id"] = topics

    models_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(models_dir / "bertopic"), serialization="pickle")

    df[["debate_id", "seq_index", "topic_id"]].to_parquet(processed_dir / "topics.parquet", index=False)

    topic_ids = topic_model.get_topic_info()["Topic"].tolist()
    print("Computing per-topic coherence (c_v)...")
    coherence = compute_coherence_per_topic(topic_model, texts, topic_ids)
    write_taxonomy(topic_model, df, topics, coherence, processed_dir / "topic_taxonomy.csv")

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers_after = sum(1 for t in topics if t == -1)
    print(f"Topics found: {n_topics}")
    print(f"Outliers before reassignment: {n_outliers_before} ({100 * n_outliers_before / len(topics):.1f}%)")
    print(f"Outliers after reassignment: {n_outliers_after} ({100 * n_outliers_after / len(topics):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    run(Path(args.config), Path(args.processed_dir), Path(args.models_dir))
