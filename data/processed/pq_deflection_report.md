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
- Transfer references found: 1076 (1041 matched to a known header)
- PQs with a resolved asker party: 11240 (90.5%)
- PQs excluded (no match or surname collision): 1186 (9.5%)
- Transferred PQs among resolved: 893 (7.9%)

## Headline: no evidence of party-conditional deflection

Transfer rate is **7.77%** for government-asked PQs and **8.01%** for opposition-asked PQs. Fisher's exact test: odds ratio=1.034, p=0.694. Logistic regression confirms this: the opposition coefficient is 0.0335 (p=0.673), not significant.

### Contingency table

| Party | Not transferred | Transferred | Transfer rate |
|---|---|---|---|
| government | 2767 | 233 | 7.77% |
| opposition | 7580 | 660 | 8.01% |

### Robustness check: interaction with Policy-topic status

H2 was originally framed around sensitive *policy* questions specifically, not procedural ones, so the party effect is also tested conditional on whether the question falls in one of the 50 labeled Policy domains.

| Party | Policy topic | N | Transferred | Rate |
|---|---|---|---|---|
| government | False | 1096 | 92 | 8.39% |
| government | True | 1904 | 141 | 7.41% |
| opposition | False | 2426 | 230 | 9.48% |
| opposition | True | 5814 | 430 | 7.40% |

Neither the main party effect nor the party x policy-topic interaction term is statistically significant in the logistic regression (interaction p=0.410).

## Caveats

- Asker resolution uses the same surname-matching method as the Linking layer's backbench-MP resolution (see party_resolution.py); collisions within an Assembly term are left unresolved rather than guessed, which is why 1186 of 12426 PQs are excluded rather than assigned a party.
- Ministers and the Prime Minister rarely ask Parliamentary Questions of their own government, so the resolved-asker population is dominated by opposition and backbench government MPs; this test is about backbench/opposition dynamics, not a full chamber comparison.
- 3% of transfer references did not match a known PQ header in the same debate (likely a header the regex missed, e.g. an unusual name format) and are not counted as transferred for any PQ - a small, non-systematic undercount of the transferred total.
