# H2 Reassessment - Individual-PQ Question Deflection

The original H2 test (see linking_layer_report.md) counted 226 pnq_transfer-
tagged utterances split across 50 topics - most topic cells had 0-2 events,
too sparse to test anything. Reading a sample of those utterances found why:
84% are the Speaker's routine end-of-Question-Time announcement, and each one
bundles a median of 3 straggler PQ numbers together - a scheduling artifact,
not 226 independent deflection decisions.

Every Parliamentary Question carries a `(No. B/xxx) <Name> (<Constituency>)`
header in the same convention Layer 0 already parses for roles, giving each
one a known asker. This lets H2 be tested as originally intended: whether a
question's asker predicts whether it gets transferred, across the full PQ
population, rather than a sparse per-topic breakdown.

## Data

- Distinct PQs identified corpus-wide: 12426
- Transfer references found: 433 (418 matched to a known header)
- PQs with a resolved asker party: 11240 (90.5%)
- PQs excluded (no match or surname collision): 1186 (9.5%)
- Transferred PQs among resolved: 363 (3.2%)

## Unadjusted party comparison

Transfer rate is **2.37%** for government-asked PQs and **3.54%** for opposition-asked PQs. Fisher's exact test: odds ratio=1.516, p=0.002. Logistic regression: the opposition coefficient is 0.4158 (p=0.002).

These tests treat PQs as independent, include withdrawn PQs in the denominator and have no controls. The adjusted, clustered analysis that the paper reports is in `h2_reanalysis_report.md`.

### Contingency table

| Party | Not transferred | Transferred | Transfer rate |
|---|---|---|---|
| government | 2929 | 71 | 2.37% |
| opposition | 7948 | 292 | 3.54% |

### Robustness check: interaction with Policy-topic status

H2 was originally framed around sensitive *policy* questions specifically, not procedural ones, so the party effect is also tested conditional on whether the question falls in one of the 50 labeled Policy domains.

| Party | Policy topic | N | Transferred | Rate |
|---|---|---|---|---|
| government | False | 1096 | 29 | 2.65% |
| government | True | 1904 | 42 | 2.21% |
| opposition | False | 2426 | 79 | 3.26% |
| opposition | True | 5814 | 213 | 3.66% |

Party x policy-topic interaction in the logistic regression: p=0.269.

## Caveats

- Asker resolution uses the same surname-matching method as the Linking layer's backbench-MP resolution (see party_resolution.py); collisions within an Assembly term are left unresolved rather than guessed, which is why 1186 of 12426 PQs are excluded rather than assigned a party.
- Ministers and the Prime Minister rarely ask Parliamentary Questions of their own government, so the resolved-asker population is dominated by opposition and backbench government MPs; this test is about backbench/opposition dynamics, not a full chamber comparison.
- 3% of transfer references did not match a known PQ header in the same debate (likely a header the regex missed, e.g. an unusual name format) and are not counted as transferred for any PQ - a small, non-systematic undercount of the transferred total.
