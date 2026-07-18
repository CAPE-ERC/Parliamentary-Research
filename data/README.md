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
