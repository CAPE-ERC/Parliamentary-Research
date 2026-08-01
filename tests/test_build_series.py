import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outcome_layer.build_series import build_chronological_debates


def test_chronological_ordering_and_unparseable_drop():
    utterances = pd.DataFrame(
        {
            "debate_id": ["d1", "d1", "d2", "d2", "d3", "d3"],
            "sitting_date": [
                "TUESDAY 10 FEBRUARY 2015",
                "TUESDAY 10 FEBRUARY 2015",
                "MONDAY 05 JANUARY 2015",
                "MONDAY 05 JANUARY 2015",
                None,
                None,
            ],
        }
    )
    debates, dropped = build_chronological_debates(utterances)

    assert len(debates) == 2
    assert len(dropped) == 1
    assert dropped["debate_id"].iloc[0] == "d3"
    # d2 (05 Jan) should sort before d1 (10 Feb)
    assert list(debates["debate_id"]) == ["d2", "d1"]
    assert list(debates["sitting_order"]) == [0, 1]


def test_grid_fill_zero_for_topics_not_discussed():
    """A topic present in one sitting and absent in another should get
    attention_share=0 (not a dropped row) for the sitting it's absent from -
    the whole point of the full grid, per the plan's design rationale."""
    from outcome_layer.build_series import build_series
    import tempfile

    utterances = pd.DataFrame(
        {
            "debate_id": ["d1", "d1", "d2"],
            "seq_index": [0, 1, 0],
            "sitting_date": [
                "TUESDAY 10 FEBRUARY 2015",
                "TUESDAY 10 FEBRUARY 2015",
                "MONDAY 05 JANUARY 2015",
            ],
            "is_stage_direction": [False, False, False],
            "role": ["unclassified_mp", "unclassified_mp", "unclassified_mp"],
        }
    )
    topics = pd.DataFrame(
        {
            "debate_id": ["d1", "d1", "d2"],
            "seq_index": [0, 1, 0],
            "predicted_label": ["Healthcare services, hospitals and medical workforce", "non_policy", "non_policy"],
        }
    )
    procedural_tags = pd.DataFrame(
        {
            "debate_id": ["d1", "d1", "d2"],
            "seq_index": [0, 1, 0],
            "chair_ruling_rule": [0, 0, 0],
            "interruption_rule": [0, 0, 0],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        utterances.to_parquet(tmp_path / "utterances.parquet")
        topics.to_parquet(tmp_path / "utterance_policy_labels_two_stage.parquet")
        procedural_tags.to_parquet(tmp_path / "procedural_tags_final.parquet")

        series = build_series(tmp_path)

    healthcare = series[series["topic"] == "Healthcare services, hospitals and medical workforce"]
    assert len(healthcare) == 2  # both sittings present, even though only discussed in one
    d1_row = healthcare[healthcare["debate_id"] == "d1"].iloc[0]
    d2_row = healthcare[healthcare["debate_id"] == "d2"].iloc[0]
    assert d1_row["attention_share"] > 0
    assert d2_row["attention_share"] == 0
