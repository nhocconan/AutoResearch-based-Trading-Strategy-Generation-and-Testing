# Independent Verification Guide

This document enables a **separate AI or human analyst** to independently verify
the strategies and results produced by the autoresearch system. No access to the
research agent or its code is needed — only the data, strategies, and this guide.

---

## 1. Data Access

### 1.1 Price Data (OHLCV)

**Location:** `data/processed/klines/{SYMBOL}/{TIMEFRAME}.parquet`

| Symbol | Timeframes | Rows (1h) | Date Range |
|--------|-----------|-----------|------------|
| BTCUSDT | 1m, 5m, 15m, 1h, 4h, 1d | 45,240 | 2021-01-01 → 2026-02-28 |
| ETHUSDT | 1m, 5m, 15m, 1h, 4h, 1d | 45,240 | 2021-01-01 → 2026-02-28 |
| SOLUSDT | 1m, 5m, 15m, 1h, 4h, 1d | 45,240 | 2021-01-01 → 2026-02-28 |

**Format:** Apache Parquet (Snappy compression)

**Schema:**
```
open_time                 datetime64[ms, UTC]   # Bar open timestamp
open                      float64               # Open price (USDT)
high                      float64               # High price
low                       float64               # Low price
close                     float64               # Close price
volume                    float64               # Base asset volume (e.g., BTC)
close_time                datetime64[ms, UTC]   # Bar close timestamp
quote_volume              float64               # Quote volume (USDT)
trades                    int64                 # Number of trades in bar
taker_buy_volume          float64               # Taker buy volume (base)
taker_buy_quote_volume    float64               # Taker buy volume (USDT)
```

**How to load:**
```python
import pandas as pd

# Load 1-hour BTCUSDT data
df = pd.read_parquet("data/processed/klines/BTCUSDT/1h.parquet")

# Split train/test
train = df[df["open_time"] < "2025-01-01"]  # 2021-01-01 to 2024-12-31
test = df[df["open_time"] >= "2025-01-01"]   # 2025-01-01 onwards
```

**Data source:** [Binance Public Data](https://data.binance.vision/) — USDT-M Futures monthly klines. Downloaded and processed by `prepare.py` (no modifications to raw data, only format conversion).

### 1.2 Funding Rate Data

**Location:** `data/processed/funding/{SYMBOL}/funding_rate.parquet`

**Schema:**
```
calc_time                 datetime64[ms, UTC]   # Funding calculation timestamp
funding_interval_hours    int64                 # Interval (always 8)
last_funding_rate         float64               # Rate (e.g., 0.0001 = 0.01%)
```

**Frequency:** Every 8 hours (00:00, 08:00, 16:00 UTC)

**How to load:**
```python
funding = pd.read_parquet("data/processed/funding/BTCUSDT/funding_rate.parquet")
```

### 1.3 Strategy Code

**Location:** `strategies/{strategy_name}.py`

Each file is a standalone Python module with:
```python
name = "strategy_name"           # Identifier
timeframe = "1h"                 # Primary timeframe
leverage = 1.0                   # Leverage multiplier

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Input: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    Output: numpy array of same length, values in [-1.0, 1.0]
            Positive = long, negative = short, 0 = flat
            The VALUE is the position size (0.35 = 35% of capital)
    """
```

### 1.4 Experiment Results

**Location:** `results.tsv` (tab-separated)

**Columns:**
```
commit      - Git commit hash
strategy    - Strategy name
symbol      - Trading pair (BTCUSDT/ETHUSDT/SOLUSDT)
sharpe      - Annualized Sharpe ratio
return_pct  - Total return %
cagr_pct    - Compound annual growth rate %
max_dd_pct  - Maximum drawdown % (negative)
win_rate    - Win rate %
profit_factor - Gross profit / gross loss
trades      - Number of completed trades
sortino     - Sortino ratio
calmar      - Calmar ratio
status      - "keep" or "discard"
description - Experiment description
period      - "train" or "test"
```

---

## 2. How to Independently Verify a Strategy

### 2.1 Quick Verification (use existing engine)

```python
from backtest import run_strategy_backtest
from evaluate import compute_metrics, print_metrics

# Pick a strategy to verify
strategy_path = "strategies/mtf_hma_rsi_atr_dynamic_tp_v1.py"

for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    for period in ["train", "test"]:
        result = run_strategy_backtest(
            strategy_path=strategy_path,
            symbol=symbol,
            period=period,
        )
        metrics = compute_metrics(result)
        print_metrics(metrics, f"{symbol} {period}")

        # Access individual trades:
        for trade in result.trades[:5]:
            print(f"  {trade.entry_time} → {trade.exit_time} | "
                  f"{'LONG' if trade.direction == 1 else 'SHORT'} | "
                  f"Entry=${trade.entry_price:.2f} Exit=${trade.exit_price:.2f} | "
                  f"PnL=${trade.pnl:.2f} ({trade.pnl_pct*100:.3f}%) | "
                  f"Fee=${trade.fee_cost:.2f}")

        # Access equity curve:
        # result.equity_curve  — numpy array, one value per bar
        # result.returns       — numpy array, per-bar returns
```

### 2.2 Full Independent Verification (write your own backtest)

If you don't trust the backtest engine, here's how to replicate from scratch:

```python
import pandas as pd
import numpy as np
import importlib.util

# 1. Load strategy
spec = importlib.util.spec_from_file_location("strat", "strategies/mtf_hma_rsi_atr_dynamic_tp_v1.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 2. Load price data
prices = pd.read_parquet(f"data/processed/klines/BTCUSDT/{mod.timeframe}.parquet")
train = prices[prices["open_time"] < "2025-01-01"].reset_index(drop=True)

# 3. Generate signals
signals = mod.generate_signals(train)

# 4. YOUR OWN backtest logic:
initial_capital = 10000.0
taker_fee_pct = 0.04 / 100    # 0.04% per side
slippage_pct = 0.01 / 100     # 0.01% per side
cost_per_side = taker_fee_pct + slippage_pct  # 0.05% per side

equity = initial_capital
position = 0.0
entry_price = 0.0

# Signal at bar t → fill at bar t+1 open (CRITICAL: 1-bar delay)
delayed_signals = np.zeros(len(signals))
delayed_signals[1:] = signals[:-1]

for i in range(1, len(train)):
    bar_open = train["open"].iloc[i]
    bar_close = train["close"].iloc[i]
    target = delayed_signals[i]

    # Position change
    change = target - position
    if abs(change) > 1e-8:
        # Cost on the change amount
        fee = abs(change) * cost_per_side * mod.leverage * equity
        equity -= fee

    # PnL on current position
    if abs(position) > 1e-8:
        price_return = (bar_close - bar_open) / bar_open
        equity += equity * position * price_return * mod.leverage

    position = target

print(f"Final equity: ${equity:.2f}")
print(f"Return: {(equity / initial_capital - 1) * 100:.1f}%")
```

### 2.3 Verify No Look-Ahead Bias

**Critical check:** At bar index `i`, `generate_signals()` must only use `prices.iloc[:i+1]`.

Method 1 — Incremental test:
```python
# Run signals on first N bars, then N+1 bars — signal[N-1] must not change
signals_full = mod.generate_signals(train)
for test_len in [1000, 2000, 5000, 10000]:
    signals_partial = mod.generate_signals(train.iloc[:test_len])
    # Signal at index test_len-1 should be identical
    assert abs(signals_partial[-1] - signals_full[test_len - 1]) < 1e-8, \
        f"Look-ahead detected at bar {test_len - 1}!"
print("No look-ahead detected")
```

Method 2 — Code inspection:
```python
# Check for forbidden patterns
import re
code = open("strategies/mtf_hma_rsi_atr_dynamic_tp_v1.py").read()
assert not re.search(r'\.shift\s*\(\s*-', code), "Negative shift = look-ahead!"
assert not re.search(r'prices\.iloc\[.*i\s*\+', code), "Future index = look-ahead!"
assert not re.search(r'prices\[.*i\s*\+', code), "Future index = look-ahead!"
print("Code patterns OK")
```

### 2.4 Verify Fee Calculation

The backtest charges fees on **BOTH sides** (entry AND exit):

```
Entry: position goes from 0 → 0.35
  Fee = 0.35 × (0.04% + 0.01%) × leverage × equity = 0.35 × 0.05% × equity

Exit: position goes from 0.35 → 0
  Fee = 0.35 × (0.04% + 0.01%) × leverage × equity = 0.35 × 0.05% × equity

Round trip cost = 2 × 0.35 × 0.05% × equity = 0.035% of equity per round trip
```

With $10,000 equity and position size 0.35: fee per side ≈ $1.75, round trip ≈ $3.50.

Funding rate is applied every 8 hours to open positions:
```
funding_cost = position_size × funding_rate × leverage × equity
```

---

## 3. What to Verify

### 3.1 Checklist

- [ ] **Signal generation produces valid output** — array of same length as prices, values in [-1, 1]
- [ ] **No look-ahead bias** — signals at bar `i` don't change when adding more data after `i`
- [ ] **Fill delay enforced** — signal at bar `t` fills at bar `t+1` open price
- [ ] **Fees charged both sides** — entry AND exit each incur 0.05% cost
- [ ] **Funding rates applied** — every 8h to open positions, from Binance historical data
- [ ] **Train/test separation** — strategy was developed on 2021-2024 data, test on 2025+
- [ ] **Results match** — your independent backtest matches reported Sharpe/Return/DD within 1%
- [ ] **Consistent across symbols** — works on BTC, ETH, AND SOL (not just one)
- [ ] **Drawdown acceptable** — max DD > -50% on all symbols

### 3.2 Red Flags to Watch For

1. **Sharpe > 10 with very low DD** — may indicate strategy has very few trades in certain market regimes, or exploits a data artifact. Check trade distribution over time.
2. **SOL-only performance** — SOL had 100x rally 2021-2024. A strategy that only works on SOL is unreliable.
3. **Very short trade duration** — trades lasting only 1-2 bars on 15m timeframe may be noise-fitting.
4. **Position size > 0.40** — violates risk rules. Check `max(abs(signals))`.
5. **Trades clustered in one period** — all trades in 2021 bull market = not robust.
6. **Strategy uses `open` column in signals** — if `prices["open"]` is used for signal generation, this could be subtle look-ahead since the signal fills at the same bar's open.

### 3.3 Comparing Results

Your independently computed metrics should match within small tolerance:
```python
# Your result vs reported
assert abs(your_sharpe - reported_sharpe) < 0.05, "Sharpe mismatch"
assert abs(your_return - reported_return) / max(1, abs(reported_return)) < 0.02, "Return mismatch >2%"
assert abs(your_dd - reported_dd) < 1.0, "Max DD mismatch"
```

Small differences are expected from:
- Floating point precision
- Funding rate interpolation method
- Edge cases at period boundaries

---

## 4. Environment Setup

```bash
# Python 3.10+
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow pyyaml

# Verify data is present
python3 -c "
import pandas as pd
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_parquet(f'data/processed/klines/{sym}/1h.parquet')
    print(f'{sym}: {len(df)} rows, {df[\"open_time\"].min()} → {df[\"open_time\"].max()}')
"
```

No API keys or LLM access required for verification. All data is local.

---

## 5. File Map for Auditor

```
data/processed/
├── klines/
│   ├── BTCUSDT/
│   │   ├── 1m.parquet    (152MB, ~2.7M rows)
│   │   ├── 5m.parquet    (33MB)
│   │   ├── 15m.parquet   (13MB)
│   │   ├── 1h.parquet    (3.7MB, 45K rows)
│   │   ├── 4h.parquet    (936KB)
│   │   └── 1d.parquet    (160KB)
│   ├── ETHUSDT/           (same structure)
│   └── SOLUSDT/           (same structure)
└── funding/
    ├── BTCUSDT/funding_rate.parquet  (5,655 rows, every 8h)
    ├── ETHUSDT/funding_rate.parquet
    └── SOLUSDT/funding_rate.parquet

strategies/                     # 83 strategy .py files
results.tsv                     # All experiment results
backtest.py                     # Backtest engine (immutable, auditable)
evaluate.py                     # Metrics computation (immutable, auditable)
validator.py                    # Compliance checker
config.yaml                     # Configuration (fee rates, date ranges)
```

---

## 6. Contact & Feedback

After verification, report findings to the project owner. Key questions:

1. Do your independent Sharpe/Return/DD numbers match the reported values?
2. Did you find any look-ahead bias in the strategy code?
3. Are the trades realistic (reasonable entry/exit prices, durations)?
4. Is the fee model correct (0.04% taker + 0.01% slippage per side, both sides)?
5. Any concerns about overfitting to train period?

---

*Last updated: 2026-03-21*
