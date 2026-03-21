# Architecture

## Overview

This system is an autonomous trading strategy research platform inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An LLM agent iteratively develops, backtests, and improves trading strategies for BTC/ETH/SOL USDT-M perpetual futures on Binance.

The key difference from a typical script-based approach: **the LLM agent writes the strategies**, evaluates results, learns from failures, and runs 24/7 without human intervention. See [README.md](../README.md) for the full overview.

## Design Principles

1. **Radical simplicity** — Flat file structure, minimal dependencies
2. **Honest simulation** — No lookahead, realistic costs, funding rates. See [Backtesting Rules](backtesting-rules.md)
3. **Immutable infrastructure** — Data pipeline (`prepare.py`), backtest engine (`backtest.py`), and metrics (`evaluate.py`) are fixed. The agent cannot cheat by modifying evaluation.
4. **Single mutable file** — The agent only edits `strategy.py`
5. **Ratcheting progress** — Only improvements survive. The codebase monotonically improves.
6. **Position sizing first** — Signal magnitude controls risk. Max 0.40 (40% of capital). This is the #1 lesson learned.

## How AI is Used

```
program.md (human-written knowledge base, 200+ lines of trading strategies)
    ↓
agent_research.py (orchestrator)
    ↓
LLM API (qwen3.5-plus / Claude / Gemini)
    ├── Reads: current strategy.py, experiment history, failure reasons
    ├── Generates: new strategy.py with hypothesis, indicators, entry/exit logic
    └── Learns: avoids repeating failed approaches, follows phased exploration
    ↓
strategy.py (THE ONLY FILE THE LLM EDITS)
    ↓
backtest.py + evaluate.py (IMMUTABLE — the LLM cannot cheat)
    ↓
results.tsv + strategies/ (all results logged, good strategies saved)
    ↓
dashboard.py (live monitoring, trade-by-trade verification)
```

The AI's role is strictly **strategy generation and hypothesis formation**. It does NOT evaluate its own results, modify the backtest engine, or access test data during optimization.

## Research Loop (Karpathy Pattern)

1. **Hypothesize** — LLM picks from [knowledge base](../program.md): trend following, mean reversion, momentum, multi-timeframe, ensemble
2. **Implement** — LLM writes `strategy.py` with proper position sizing (0.20-0.35), discrete signal levels, and vectorized calculations
3. **Validate** — [Compliance checker](../validator.py) scans for lookahead bias, invalid timeframes, leverage violations
4. **Backtest** — Engine runs on BTCUSDT/ETHUSDT/SOLUSDT train data (2021-2024) with 120s timeout per symbol
5. **Evaluate** — Auto-reject if: DD > -50%, trades < 10, Sharpe ≤ 0
6. **Keep or discard** — All strategies with Sharpe > 0 are saved. Best Sharpe updates the active `strategy.py`.
7. **Test** — Kept strategies also run on test period (2025+) for out-of-sample validation
8. **Log** — Results appended to `results.tsv`, code saved to `strategies/`, docs to `docs/strategies/`
9. **Repeat forever** — No max limit. Runs 24/7.

## Phased Exploration

The agent follows a structured research plan (see [program.md](../program.md)):

| Phase | Experiments | Focus |
|-------|------------|-------|
| 1 | 1-20 | Signal combinations: trend + entry timing + risk filter |
| 2 | 21-50 | Optimize best combinations: parameters, timeframes, sizing |
| 3 | 51-100 | Ensembles & regime detection |
| 4 | 100+ | Risk management optimization |

## Data Flow

```
Binance Public Data → prepare.py → Parquet files (data/processed/)
                                         ↓
strategy.py → generate_signals(prices) → signals array
                                         ↓
                     backtest.py → equity_curve, trades, returns
                                         ↓
                     evaluate.py → Sharpe, DD, WR, PF, etc.
                                         ↓
                     results.tsv + strategies/ + docs/
                                         ↓
                     dashboard.py → web UI with charts & trade detail
```

## Cost Model

See [Backtesting Rules](backtesting-rules.md) for full details.

| Cost | Value | Source |
|------|-------|--------|
| Taker fee | 0.04% per side | Binance USDT-M futures |
| Slippage | 0.01% per side | Conservative estimate |
| Round trip | 0.10% total | Fee + slippage both sides |
| Funding | Every 8h | Binance historical data |
| Fill delay | 1 bar | Signal at t → fill at t+1 open |

## Last Updated
2026-03-21
