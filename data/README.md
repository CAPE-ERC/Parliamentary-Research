# Data

This folder is intentionally kept out of version control (see root `.gitignore`) —
parliamentary transcript data can be large and/or subject to redistribution
restrictions from the source legislature.

- `raw/` — original debate transcripts as sourced (one file per sitting/debate).
  Document provenance (source, date range, retrieval method) here once populated.
- `processed/` — cleaned, turn-segmented, and labeled datasets produced by the
  `src/*_layer` pipelines. Regeneratable from `raw/` + `src/`.
- `external/` — reference data not from the debate corpus itself: standing
  orders text, speaker/party registries, sitting calendars, etc.

Only `.gitkeep` placeholders are tracked so the folder structure ships with the
repo; add a short provenance note here when real data is added.

## Known gap: speaker/party registry

`external/speaker_registry_TEMPLATE.csv` is an empty template
(`assembly_term, surname_key, full_name, party, constituency, start_date,
end_date`). Government ministers and chair officers can be derived per-debate
from each PDF's own Cabinet/Officers front matter (see
`src/preprocessing/roles.py`), but backbench MPs have no party info anywhere
in the Hansard PDFs themselves. Populating this registry (Mauritius National
Assembly membership by term - 2014/2019/2024 elections span the corpus) is a
follow-up data-sourcing task, not yet done.

## Tracked exception: `processed/topic_taxonomy.csv`

Everything else under `processed/` is gitignored as regeneratable output, but
`topic_taxonomy.csv` is deliberately tracked: it's the team's shared working
document for assigning human-readable policy-domain names to the 199
BERTopic clusters (`label` column, currently blank) before Stage 2 (the
fine-tuned classifier) trains on those labels. Produced by
`src/topic_layer/train_bertopic.py`, which now fixes `random_state=42` so
re-running reproduces the same topic assignments this file corresponds to.
Includes a `coherence_c_v` column (gensim, c_v measure) as a triage aid -
note that pure-language clusters (e.g. French/Kreol function words) can score
high on coherence without being real policy topics, while low coherence
reliably flags noise/procedural-attribution clusters (MP surname co-mentions)
worth dropping or merging rather than naming.

## Tracked exception: `processed/topic_taxonomy_labeled.xlsx`

The completed human review of the above: all 199 topics labeled
(`label`, `label_type` - Policy/Governance/Procedure/Procedure-Noise/
Discourse-Noise/Language-Noise, `review_confidence` - High/Medium/Low,
`human_review_note`), plus a `Label Guide` sheet (the labelling protocol
and decision rules per type) and `QA Summary` sheet (totals and an
interpretive note: procedural/noise clusters are retained deliberately -
excluded from policy-attention share, but analytically useful for the
Procedural layer's floor-control/interruption/ruling signal). This is the
source of truth for Stage 2 (fine-tuned classifier) training labels: the
50 `Policy`-type topics become distinct classes, everything else collapses
to a single non-policy catch-all class.
