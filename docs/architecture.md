# Architecture

## Overview

This system is an autonomous trading strategy research platform inspired by
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An LLM
agent iteratively develops, backtests, and improves trading strategies for
cryptocurrency perpetual futures.

## Design Principles

1. **Radical simplicity** - Flat file structure, minimal dependencies
2. **Honest simulation** - No look-ahead bias, realistic costs, funding rates
3. **Immutable infrastructure** - Data pipeline and backtest engine are fixed
4. **Single mutable file** - The agent only edits `strategy.py`
5. **Memory efficient** - Designed for 16GB M1 Mac with Parquet storage
6. **Multi-provider LLM** - Supports OpenAI, Anthropic, and Gemini APIs

## File Structure

```
llm-trading-research/
├── CLAUDE.md              # Agent rules (immutable)
├── config.yaml            # Configuration (immutable during research)
├── program.md             # Research protocol for the LLM agent
│
├── prepare.py             # [IMMUTABLE] Data download + preprocessing
├── backtest.py            # [IMMUTABLE] Backtesting engine
├── evaluate.py            # [IMMUTABLE] Performance metrics
├── llm_client.py          # [IMMUTABLE] Multi-provider LLM client
│
├── strategy.py            # [MUTABLE] Current strategy under test
├── run_research.py        # Research experiment runner
├── results.tsv            # Experiment log (append-only)
│
├── data/
│   ├── raw/               # Downloaded Binance zip files
│   └── processed/         # Parquet files for fast loading
│
├── strategies/            # Saved successful strategies
├── reports/               # Generated analysis reports
│
└── docs/
    ├── architecture.md    # This file
    ├── data-pipeline.md   # Data format and pipeline docs
    ├── backtesting-rules.md # Simulation rules
    └── strategies/        # Strategy documentation
```

## Data Flow

```
Binance API → raw zips → Parquet files → backtest.py → evaluate.py → results.tsv
                                              ↑                           ↓
                                         strategy.py ← LLM Agent ← analysis
```

## Research Loop

Following the autoresearch pattern:

1. **Hypothesize** - LLM agent decides what to try
2. **Implement** - Agent edits `strategy.py`, commits
3. **Execute** - `python run_research.py` backtests all symbols
4. **Evaluate** - Compare Sharpe, return, drawdown vs current best
5. **Decide** - Keep (better) or discard (worse/crash)
6. **Log** - Append to `results.tsv`
7. **Repeat** - Never stop

## Train/Test Split

- **Train: 2021-01-01 to 2024-12-31** - Free for development and optimization
- **Test: 2025-01-01 to present** - Final evaluation only, never optimize on this

## Execution Model

Signal at bar `t` → order fills at bar `t+1` open price:
- Prevents look-ahead bias
- Accounts for reaction time
- Enforced by the backtest engine (signal array shifted by 1)

## Cost Model

Per trade (one side):
- Taker fee: 0.04% (Binance futures)
- Slippage: 0.01% (conservative estimate)
- Total: 0.05% per side, 0.10% round trip

Funding rate: applied every 8 hours to open positions (from Binance data).

## Last Updated
2026-03-20
