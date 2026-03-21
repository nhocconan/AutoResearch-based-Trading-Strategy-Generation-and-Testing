# LLM Trading Research - Project Rules

## Project Overview
Autonomous trading strategy research system inspired by Karpathy's autoresearch.
An LLM agent iteratively develops, tests, and improves trading strategies for
BTC/ETH/SOL USDT-margined perpetual futures using Binance historical data.

## Evaluation Flow (STRICTLY ENFORCED)

```
1. LLM generates strategy.py
2. Validator checks: no lookahead, no manual MTF, no get_htf_data in loop
3. PER-SYMBOL independent evaluation:
   For each symbol (BTC, ETH, SOL):
     a. Train backtest → Sharpe > 0 AND trades >= 5? → train PASS
     b. If train FAIL → skip test for this symbol, try next symbol
     c. If train PASS → run test → Sharpe > 0 AND trades >= 3? → KEEP for this symbol
4. Prefix look-ahead test (if any symbol kept)
5. Strategy is KEPT if at least 1 symbol passes BOTH train AND test
```

**Per-symbol evaluation:** Each symbol is independent. BTC can fail while
ETH passes. A strategy is kept if it works on ANY symbol. This reflects
reality: BTC, ETH, SOL have different market characteristics.

**Early discard within symbol:** If train fails for a symbol, skip its test.
But ALWAYS try all 3 symbols — don't stop at first failure.

**0-trade strategies are ALWAYS discarded.** Sharpe=0.000 with 0 trades is NOT a pass.

## Data

- Source: Binance Public Data (futures/um), downloaded via `prepare.py` + `update_data.py`
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Timeframes: 5m, 15m, 30m, 1h, 4h, 6h, 12h, 1d (1m excluded — too noisy)
- Format: Parquet files in `data/processed/`
- Date range: 2021-01-01 to present (updated with `update_data.py`)
- Train period: 2021-01-01 to 2024-12-31
- Test period: 2025-01-01 to present (~15 months)
- ALL data is real Binance data. NEVER resample to create timeframes.

## Backtest Engine

- Signal at bar `t` → fill at bar `t+1` open price (enforced by engine)
- Warmup: engine loads ALL historical data, runs signals on full history,
  then trims PnL to period. This ensures indicators are warm on test period.
- Taker fee: 0.04% per side + Slippage: 0.01% per side = 0.10% round trip
- Funding rate: applied every 8 hours from Binance historical data
- Leverage: 1.0 (no leverage unless explicitly justified)
- Timeout: 90 seconds per symbol (multiprocessing hard kill)

## Strategy Code Rules (see STRATEGY_RULES.md for full detail)

### Position Sizing
- Signal value = position size. signal=0.30 means 30% of capital.
- MAX magnitude: 0.40. NEVER use 1.0.
- Use DISCRETE levels: 0.0, ±0.15, ±0.30 to minimize fee churn.

### Multi-Timeframe (MTF)
- MUST use `mtf_data.get_htf_data()` to load higher timeframe data
- Call `get_htf_data()` ONCE before the loop, use aligned arrays inside
- NEVER call `get_htf_data()` inside a for/while loop (loads Parquet file each call)
- NEVER use `i // N` manual index mapping (uses unclosed HTF bars = look-ahead)
- NEVER use `pd.date_range()` or `.resample()` to create HTF data
- `align_htf_to_ltf()` auto-shifts by 1 HTF bar to only use COMPLETED bars

### No Look-Ahead Bias
- At bar index i, only use `prices.iloc[:i+1]`
- No `.shift(-n)` (negative shift)
- No future index access
- Prefix look-ahead test: signals on N bars must match signals on N+M bars at index N-1

### Must Generate Trades
- Train: ≥ 5 trades per symbol (≥ 10 average)
- Test: ≥ 3 trades per symbol
- If entry conditions are too strict, LOOSEN them

## Mutable Files
- `strategy.py` - Current strategy under test (LLM edits this)
- `results.tsv` - Experiment log (append-only)
- `strategies/` - Saved strategy code (all with Sharpe > 0 on train)

## Key Files
- `STRATEGY_RULES.md` - Detailed code rules for LLM (with examples)
- `program.md` - Research protocol and strategy knowledge base
- `mtf_data.py` - Multi-timeframe data loader (MUST use for HTF)
- `validator.py` - Compliance checker (AST + regex)
- `agent_research.py` - Main research loop
- `backtest.py` - Backtesting engine
- `evaluate.py` - Performance metrics
- `dashboard.py` - Web dashboard (http://localhost:8888)
- `revalidate.py` - Rerun all saved strategies
- `update_data.py` - Download latest daily data from Binance
- `run.sh` - Convenience script (--watchdog for auto-restart)

## Running
```bash
./run.sh --all          # Full setup: data + dashboard + research
./run.sh --watchdog     # Run with auto-restart watchdog (recommended)
./run.sh --status       # Check progress
./run.sh --stop         # Stop everything
python update_data.py   # Download latest daily data
python revalidate.py    # Rerun all saved strategies
```
