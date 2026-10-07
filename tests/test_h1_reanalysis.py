import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linking_layer.h1_reanalysis import flag_window, speaker_key, tost_p


def _frame(debates, events):
    return pd.DataFrame({
        "debate_id": debates,
        "seq_index": range(len(debates)),
        "event": events,
    })


def test_flag_window_looks_only_forward():
    df = _frame(["a"] * 5, [False, False, True, False, False])
    assert flag_window(df, "event", 3).tolist() == [True, True, False, False, False]


def test_flag_window_respects_window_length():
    df = _frame(["a"] * 5, [False, False, False, False, True])
    assert flag_window(df, "event", 1).tolist() == [False, False, False, True, False]
    assert flag_window(df, "event", 3).tolist() == [False, True, True, True, False]


def test_flag_window_does_not_cross_debates():
    df = _frame(["a", "a", "b", "b"], [False, False, True, False])
    assert flag_window(df, "event", 3).tolist() == [False, False, False, False]


def test_speaker_key_normalises_whitespace_and_case():
    assert speaker_key("Mr  Bodha ") == "mr bodha"
    assert speaker_key(None) is None


def test_tost_rejects_only_when_estimate_is_well_inside_bounds():
    assert tost_p(0.0, 0.01, -0.05, 0.05) < 0.001
    assert tost_p(0.04, 0.01, -0.05, 0.05) > 0.05
    assert tost_p(0.0, 0.05, -0.05, 0.05) > 0.05
