import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outcome_layer.granger_test import per_topic_granger, pooled_panel_model, MIN_NONZERO_CONFLICT_SITTINGS


def _make_series(topic: str, n: int, conflict_rate, attention_share) -> pd.DataFrame:
    return pd.DataFrame({
        "topic": topic,
        "debate_id": [f"d{i}" for i in range(n)],
        "sitting_order": range(n),
        "sitting_date": pd.date_range("2015-01-01", periods=n, freq="D"),
        "attention_share": attention_share,
        "conflict_rate": conflict_rate,
    })


def test_topic_with_too_little_conflict_variation_is_skipped():
    n = 30
    rng = np.random.default_rng(0)
    series = _make_series(
        "quiet_topic", n,
        conflict_rate=np.zeros(n),  # zero variation -> below MIN_NONZERO_CONFLICT_SITTINGS
        attention_share=rng.uniform(0, 0.1, n),
    )
    result = per_topic_granger(series)
    assert result["skipped"].all()
    assert result["n_conflict_sittings"].iloc[0] == 0


def test_topic_with_enough_variation_produces_lag_1_to_3_rows():
    n = 60
    rng = np.random.default_rng(1)
    conflict = np.zeros(n)
    conflict[rng.choice(n, MIN_NONZERO_CONFLICT_SITTINGS + 5, replace=False)] = 0.5
    series = _make_series(
        "active_topic", n,
        conflict_rate=conflict,
        attention_share=rng.uniform(0, 0.1, n),
    )
    result = per_topic_granger(series)
    assert not result["skipped"].any()
    assert sorted(result["lag"].tolist()) == [1, 2, 3]
    assert result["p_value"].between(0, 1).all()


def test_pooled_panel_model_shifts_attention_share_forward_by_topic():
    series = pd.concat([
        _make_series("t1", 5, conflict_rate=[0, 1, 0, 1, 0], attention_share=[0.1, 0.2, 0.3, 0.4, 0.5]),
        _make_series("t2", 5, conflict_rate=[1, 0, 1, 0, 1], attention_share=[0.5, 0.4, 0.3, 0.2, 0.1]),
    ])
    _, panel_df = pooled_panel_model(series)

    # last sitting per topic has no "next" row and must be dropped
    assert len(panel_df) == 8
    t1_first = panel_df[(panel_df["topic"] == "t1") & (panel_df["sitting_order"] == 0)].iloc[0]
    assert t1_first["attention_share"] == 0.1
    assert t1_first["attention_share_next"] == 0.2
