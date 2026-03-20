# LLM Trading Research - Agent Rules

## Project Overview
Autonomous trading strategy research system inspired by Karpathy's autoresearch.
An LLM agent iteratively develops, tests, and improves trading strategies for
BTC/ETH/SOL USDT-margined perpetual futures using Binance historical data.

## Critical Rules

### Immutable Files (NEVER modify these)
- `prepare.py` - Data download and preprocessing
- `backtest.py` - Backtesting engine
- `evaluate.py` - Evaluation metrics
- `config.yaml` - Configuration (unless adding a new LLM provider)

### Mutable Files (Agent edits these)
- `strategy.py` - The current strategy being tested
- `results.tsv` - Append experiment results only

### No Look-Ahead Bias
- Signal at bar `t` → order filled at bar `t+1` open price
- Strategy MUST NOT use any data from time > t when generating signal at time t
- All indicators must use only past data (no future data leakage)
- Train period: 2021-01-01 to 2024-12-31
- Test period: 2025-01-01 to present
- NEVER optimize on test data. Test is for final evaluation only.

### Trading Simulation Honesty
- Taker fee: 0.04% per side
- Slippage: 0.01% per side
- Total cost: 0.05% per side (0.10% round trip)
- Funding rate: applied every 8 hours to open positions
- Leverage: configurable per strategy (default 1x, max 20x)

### Data
- Source: Binance Public Data (futures/um)
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Format: Parquet files in `data/processed/`

### Documentation
- All strategies must be documented in `docs/strategies/`
- Keep `docs/` up-to-date when making changes
- Document why a strategy works or fails, not just what it does

## Code Style
- Python 3.10+
- Use type hints
- Use numpy/pandas for vectorized operations (avoid loops for performance)
- Keep strategies simple and readable
- One strategy per experiment

## Running
```bash
# Download and prepare data
python prepare.py

# Run a single backtest
python backtest.py

# Run the research loop
python run_research.py
```
