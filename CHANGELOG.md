# Changelog

All notable changes to the honest-simulation engine and research infrastructure
are documented here. Dates use ISO 8601.

## [Unreleased] — Engine integrity & contributor infrastructure

This series hardens the project: it adds the first automated tests and CI, then
fixes two real honest-simulation bugs that the new tests exposed.

### Added
- **Test suite** (`tests/`) — 47 tests covering no-look-ahead fills, trading
  costs, performance metrics, the compliance validator, multi-timeframe
  alignment, and keep/discard rules. Runs entirely on synthetic data (no market
  data needed), so it is reproducible in CI.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — lint + validator +
  pytest + results-integrity, on Python 3.11 and 3.12.
- **`scripts/check_results_integrity.py`** — independent guard that
  `results.tsv` has no duplicate `(strategy, symbol, period)` rows.
- **This changelog** and an expanded, diagram-rich README / contributor guide.

### Fixed
- **Funding is now signed by position direction** (`backtest.py`). Previously
  funding was applied with `abs(current_position)` and accumulated with
  `abs(...)`, so **both** longs and shorts were *charged* funding. On Binance
  perps a positive funding rate means longs pay shorts — the short side should
  *receive* funding. The engine now charges `current_position * rate * leverage`
  (signed), so shorts correctly earn funding in positive-funding regimes and
  longs pay. Proven by `tests/test_engine_integrity.py`.
- **Trading-cost config reconciled with the documented model** (`config.yaml`).
  The config charged `0.025%` taker + `0%` slippage = **0.05% round trip**,
  while the README, `CLAUDE.md`, `docs/`, and the validator all stated **0.10%
  round trip** (`0.04%` taker + `0.01%` slippage). Because the engine fills at
  the next bar's open via an implicit market order (taker on both sides), the
  conservative documented `0.10%` is the honest number. Config now matches, and
  a test pins config ⇆ docs together so they cannot drift again.

### Removed
- Stray `tmpoa30fwld.py` scratch file committed at the repo root.

### Note on historical results
`results.db` / `results.tsv` rows produced **before** these fixes were computed
with the old funding sign and the lower `0.05%` round-trip cost. They remain in
the log for provenance. Re-running `python revalidate.py` recomputes saved
strategies under the corrected, stricter engine; treat pre-fix and post-fix
rows as different cost/funding regimes when comparing.
