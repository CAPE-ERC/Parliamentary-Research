import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topic_layer.assignment_confidence import score_assignments
from validation.score_annotations import weighted_pr, wilson


def test_weighted_pr_uses_weights():
    truth = pd.Series([True, True, False, False])
    pred = pd.Series([True, False, True, False])
    unweighted = weighted_pr(truth, pred, pd.Series([1.0, 1.0, 1.0, 1.0]))
    assert unweighted["precision"] == 0.5 and unweighted["recall"] == 0.5
    weighted = weighted_pr(truth, pred, pd.Series([3.0, 1.0, 1.0, 1.0]))
    assert weighted["precision"] == 0.75
    assert weighted["recall"] == 0.75


def test_wilson_interval_brackets_proportion():
    lo, hi = wilson(8, 10)
    assert lo < 0.8 < hi
    assert 0 <= lo and hi <= 1


def test_score_assignments_flags_distant_outliers():
    embeddings = np.array([
        [1.0, 0.0], [0.99, 0.05], [0.98, -0.05],  # topic 0 core
        [0.0, 1.0], [0.05, 0.99], [-0.05, 0.98],  # topic 1 core
        [0.6, 0.8],                               # outlier assigned to topic 1, far from its core
    ])
    topic_ids = np.array([0, 0, 0, 1, 1, 1, 1])
    was_outlier = np.array([False] * 6 + [True])
    scores = score_assignments(embeddings, topic_ids, was_outlier)
    assert bool(scores.loc[6, "low_confidence"])
    assert not scores.loc[:5, "low_confidence"].any()
    assert scores.loc[6, "margin"] < scores.loc[3, "margin"]
