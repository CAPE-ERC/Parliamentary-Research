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
