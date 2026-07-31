# Procedural Layer - LSTM Training Report

- Vocabulary size: 20002
- Training examples: 190410 (held-out: 21157)
- **No human-annotated gold set** - trained on the rule-tagger's silver labels. The metrics below measure agreement with those rules (confirms training worked), not independent real-world accuracy.

## Held-out agreement with rule-tagger labels

| Tag | Precision | Recall | F1 |
|---|---|---|---|
| chair_ruling | 0.457 | 0.977 | 0.623 |
| interruption | 1.000 | 1.000 | 1.000 |
| withdrawal_request | 0.813 | 0.948 | 0.875 |
| so_citation | 0.549 | 0.888 | 0.678 |
| pnq_transfer | 0.788 | 0.897 | 0.839 |

## Full-corpus tag counts (combined = rule OR lstm)

| Tag | Rule count | LSTM count | Combined count |
|---|---|---|---|
| chair_ruling | 5055 | 10863 | 10885 |
| interruption | 27607 | 27611 | 27611 |
| withdrawal_request | 5706 | 6649 | 6703 |
| so_citation | 850 | 1297 | 1315 |
| pnq_transfer | 226 | 291 | 294 |

Disagreement sample (up to 20 per tag) written to procedural_disagreements_sample.csv for spot-checking.

## What manually reading the disagreement sample actually showed

Precision/recall against silver labels only tells you whether the LSTM
reproduced the rules - it doesn't tell you whether the LSTM's *extra* positives
are real. I read a sample of each tag's disagreements (rule says no, LSTM says
yes) and found genuinely different stories per tag:

- **pnq_transfer (trustworthy)**: LSTM's extra positives are real catches -
  e.g. "PQ B/892 **would** be replied by..." (the rule's regex only matched
  "**will** be replied by"; fixed after finding this - see
  `test_pnq_transfer_would_variant` in `tests/test_procedural_rules.py`), and
  PQ-withdrawal announcements ("PQ B/144...has been withdrawn") that are
  substantively the same H2 phenomenon (deflection/non-response) even though
  they don't literally say "transfer". The `combined` column is a real
  improvement over `rule` alone here.

- **withdrawal_request (mostly trustworthy, some noise)**: some genuine misses
  (chair remarks without the literal word "withdraw"), but also confusion with
  the formulaic "Table has been advised..." register shared with PQ-transfer
  announcements - a PQ-transfer utterance sometimes gets flagged as
  withdrawal_request too, since both come from the same chair-announcement
  style. Use `combined`, but don't treat it as precise.

- **chair_ruling (use `rule`, not `combined`)**: two distinct issues found.
  (1) Some extra LSTM positives are genuine misses caused by upstream Layer 0
  role-resolution gaps - e.g. a bare "Order!" that should have `role=speaker`
  but didn't get resolved that way. (2) More importantly: **the LSTM has no
  access to the `role` field at all - it only sees utterance text** - so it
  cannot learn the rule's "chair speaking" constraint. It flags MP utterances
  like "So, may I raise a point of order?" as chair_ruling just because the
  vocabulary is procedural, even though the rule correctly requires the
  *chair* to be speaking. This is a real architectural gap, not just noise -
  a future improvement would concatenate a role embedding into the LSTM's
  classifier head. For now, `rule` (precision-first, role-gated) is the
  more defensible signal for this tag; `lstm`/`combined` skews toward
  recall at a real precision cost.

- **so_citation (use `rule`, not `combined`)**: the LSTM's extra positives are
  mostly genuine false positives - generic "point of order" or formal
  parliamentary-reporting language without an actual "Standing Order N"
  citation. The LSTM appears to have learned "sounds procedurally formal"
  rather than the narrow, specific citation pattern. `rule`'s exact-phrase
  match is the safer signal here.

- **interruption**: trivial by construction (the rule *is* an exact string
  match on a stage-direction marker the LSTM easily memorizes) - perfect
  agreement is expected and not a meaningful validation of anything beyond
  "training worked."

**Practical guidance for downstream layers**: `procedural_tags_final.parquet`
keeps `{tag}_rule`, `{tag}_lstm`, and `{tag}_combined` as separate columns
specifically so this isn't hidden behind one number. For `pnq_transfer` and
`withdrawal_request`, `_combined` is reasonable. For `chair_ruling` and
`so_citation`, prefer `_rule` unless you've done your own precision check on
`_lstm` for your specific use case.