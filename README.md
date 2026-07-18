# Parliamentary Research: Procedural Power & Policy Attention

Research pipeline studying how procedural conflict (chair rulings, interruptions,
points of order) shapes policy-topic attention across a corpus of 411 parliamentary
debates.

## Analytical framework

| Layer | Task | Method | Branch |
|---|---|---|---|
| **Topic layer** | Assign a policy-domain label to every utterance/turn across the 411-debate corpus | BERTopic / fine-tuned topic classifier | [`topic-layer`](../../tree/topic-layer) |
| **Procedural layer** | Tag chair rulings, interruptions, withdrawal requests, S.O. citations, PNQ transfers | Rule-based tagging + sequence classifier (LSTM) | [`procedural-layer`](../../tree/procedural-layer) |
| **Linking layer** | Model intervention rate as a function of topic × speaker party × chair | Panel / mixed-effects regression | [`linking-layer`](../../tree/linking-layer) |
| **Outcome layer** | Test whether topic attention share in t+1 depends on procedural conflict in t | Time-series / Granger-style test on topic-share trajectories | [`outcome-layer`](../../tree/outcome-layer) |

Each layer builds on the artifacts produced by the previous one (topic labels feed
the procedural tagging context; both feed the linking-layer regression; the
linking-layer output feeds the outcome-layer time-series test).

## Repository structure

```
├── data/
│   ├── raw/          # original debate transcripts (not committed, see data/README.md)
│   ├── processed/    # cleaned/labeled intermediate datasets
│   └── external/     # reference data (standing orders, party/speaker registries)
├── src/
│   ├── topic_layer/
│   ├── procedural_layer/
│   ├── linking_layer/
│   └── outcome_layer/
├── notebooks/         # exploratory analysis notebooks
├── configs/           # pipeline configuration
├── docs/              # methodology notes, codebooks, writeups
└── tests/
```

## Branching model

- `main` — stable scaffold and integrated pipeline
- `topic-layer` — BERTopic / topic classifier development
- `procedural-layer` — procedural tagging rules + sequence classifier
- `linking-layer` — mixed-effects regression modeling
- `outcome-layer` — time-series / Granger-style outcome tests

Work on each analytical layer happens on its own branch and is merged into `main`
via pull request once validated.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## License

Released under [CC BY 4.0](LICENSE) — reuse with attribution.
