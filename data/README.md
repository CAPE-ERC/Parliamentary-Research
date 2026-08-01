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

## Provenance

- **Source:** National Assembly of Mauritius Hansard debate PDFs, mirrored from
  the team's Google Drive (`Hansard_Data`), organized by year.
- **Contents:** 405 debate transcripts across sittings from 2015–2025 in
  `raw/<year>/`, named `Debate_No_<N>_of_<year>_–_<day>_<date>.pdf` (some
  marked `(UNREVISED)` or `(REVISED)`). 6 non-debate reference PDFs (Standing
  Orders annexes, appendices, a Three-Year Strategic Plan) are kept in
  `external/` instead, since they aren't debate transcripts.
- **Sync date:** 2026-07-18, verified byte-size-identical to the Drive source.
- Not committed to git — see root `.gitignore`. Re-sync by mirroring
  `Hansard_Data` from Drive into `raw/<year>/` again if this folder is empty.

## Tracked exceptions: `external/speaker_registry.csv`, `external/party_alignment.csv`

Government ministers and chair officers are derivable per-debate from each
PDF's own Cabinet/Officers front matter (see `src/preprocessing/roles.py`),
but backbench MPs have no party info anywhere in the Hansard PDFs themselves.
`speaker_registry.csv` closes this gap with real, researched data: ~198 MP
records (name, party, constituency) for all three elections spanning the
corpus (2014/2019/2024 -> 6th/7th/8th Assembly), sourced from Wikipedia's
constituency-by-constituency results and cross-checked against the official
National Assembly site. `party_alignment.csv` separates "which party is this
MP in" (stable) from "was that party in government on date D" (time-varying)
- critically, PMSD left the governing Alliance Lepep coalition on
19 Dec 2016 (verified via web search), so government/opposition status is
date-aware per party per term, not a static per-term label. Some rows
(Rodrigues's RPO, the 8th Assembly's smaller opposition parties) are marked
`confidence: low` where the formal coalition alignment wasn't confirmed.

Matching (`src/linking_layer/party_resolution.py`) is surname-based, scoped
to the debate's Assembly term. Surname collisions where multiple registered
MPs share a surname *within the same term* (e.g. Adrien Duval and Xavier-Luc
Duval both sat in the 6th Assembly) are left unresolved rather than guessed -
see `processed/party_resolution_report.md` for the resolution rate and
collision list. `speaker_registry_TEMPLATE.csv` (the empty schema) is kept
for reference.

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

## Tracked exceptions: `processed/corpus_quality_report.md`, `processed/classifier_eval_report.md`

Small, human-readable summary docs worth keeping on GitHub for reference
(e.g. for the Methodology/Results write-up) rather than regenerating
locally each time: Layer 0 parse/role/language stats, and Stage 2's
per-class precision/recall/F1 on the held-out validation split.

## Stage 2 v2: two-stage classifier (recommended over the v1 single-stage model)

`processed/two_stage_vs_baseline_report.md` (tracked) documents a fair,
apples-to-apples comparison between the original single 51-class classifier
(`src/topic_layer/train_classifier.py`) and a two-stage pipeline
(`src/topic_layer/train_two_stage.py`: a binary Policy/non_policy gate, then
a 50-way domain classifier trained only on Policy examples). Both were
evaluated on the SAME held-out set drawn from the full, naturally-distributed
corpus (68.9% non_policy - the true prevalence), not the downsampled
training pool the original 79%/0.78-macro-F1 figure was measured on (which
was artificially skewed to ~9% non_policy and is not a trustworthy number).

On the fair comparison: single-stage 79.2% accuracy / 0.644 macro F1 vs.
two-stage **84.7% accuracy / 0.680 macro F1**. The two-stage pipeline wins
and is the recommended model going forward.

**Use `processed/utterance_policy_labels_two_stage.parquet`** (not
`utterance_policy_labels.parquet`, the v1 output) for any downstream work -
same schema (`debate_id, seq_index, predicted_label, predicted_confidence`),
159,913 rows, every utterance labeled. The v1 file and its model
(`models/topic_classifier/`) are left in place for reference/comparison but
superseded.

## Procedural layer: rule-based tags + LSTM (tracked reports)

`processed/procedural_rules_summary.md`, `processed/procedural_layer_report.md`,
and `processed/procedural_disagreements_sample.csv` are tracked (same pattern
as above). Produced by `src/procedural_layer/pipeline.py` (rule-based tagger)
and `src/procedural_layer/train_lstm.py` (BiLSTM trained on the rule tagger's
output as silver labels - no human-annotated gold set exists for this layer,
unlike the Topic layer, since no separate annotator was available).

The final output, `processed/procedural_tags_final.parquet` (not tracked,
regeneratable), keeps `{tag}_rule`, `{tag}_lstm`, and `{tag}_combined` as
**separate columns per tag** rather than one blended signal, because manual
review of the disagreement sample found the LSTM's reliability varies a lot by
tag:

- `pnq_transfer`, `withdrawal_request`: LSTM adds real value (caught a rule
  regex gap - "would be replied by" as well as "will" - and related
  PQ-withdrawal/deflection language) - `_combined` is reasonable to use.
- `chair_ruling`, `so_citation`: the LSTM has **no access to the `role`
  field** (text-only input), so it can't learn the rule's "chair must be
  speaking" constraint and over-flags MPs discussing procedure as if they
  were the chair ruling on it; `so_citation`'s extra LSTM positives are
  mostly generic "sounds procedural" false positives, not real Standing Order
  citations. **Prefer `_rule` over `_combined` for these two tags.**

See `processed/procedural_layer_report.md` for the full per-tag reasoning and
example disagreements.

## Linking layer: mixed-effects regression (tracked report)

`processed/linking_layer_report.md` (tracked) documents the H1/H2 analysis.
Panel construction (`src/linking_layer/build_panel.py`): intervention rate
per (debate x topic x party) cell, where "intervened" means a chair_ruling
or interruption (rule-tagger `_rule` columns) occurs within the next 3
utterances after a substantive MP utterance. Chair identity
(`src/linking_layer/chair_identity.py`) is the debate's presiding Speaker
surname, re-parsed from front matter and normalized to 4 distinct
officeholders across the corpus. Model: `statsmodels` `MixedLM`,
`intervention_rate ~ C(topic) * C(party)`, random intercept by chair.

**Headline finding: H1 (asymmetric enforcement) is a real null result, not
an underpowered one.** With the registry-resolved panel (13,032
opposition-attributed utterances across 4 chairs and 48 topics - a very
different sample than the 145-utterance, single-office proxy this would
have run on before the registry work), only 1 of 47 topic x opposition
interaction terms is significant at p<0.05 - at chance level for 47 tests,
and it doesn't survive Bonferroni correction. H2 (PNQ deflection by topic)
is inconclusive rather than null: `pnq_transfer` is rare enough (226 of
211,567 utterances) that most topic-level cells have 0-2 events - not
enough signal to say anything either way at this granularity.
