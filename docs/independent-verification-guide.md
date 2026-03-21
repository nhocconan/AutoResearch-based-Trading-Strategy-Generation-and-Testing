# Independent Strategy Verification Guide

You are an independent auditor. Your job: read each strategy's code, load the raw OHLCV price data, write your OWN backtest from scratch, and report whether the claimed performance is real.

Do NOT use the project's `backtest.py` — write your own to eliminate any engine bugs.

---

## Step 1: Strategies to Verify

All strategy code is in **`strategies/`** directory. Each is a standalone Python file.

**Top 10 strategies by Sharpe (train 2021-2024):**

| File | Sharpe | Max DD | Timeframe |
|------|--------|--------|-----------|
| `mtf_hma_supertrend_adx_kama_rsi_zscore_bbw_15m_v1.py` | 16.0 | -4.0% | 15m |
| `mtf_hma_supertrend_adx_kama_rsi_zscore_bbw_15m_4h_v1.py` | 13.4 | -5.1% | 15m |
| `adaptive_regime_ensemble_hma_st_rsi_adx_bbw_15m_4h_v1.py` | 13.0 | -4.1% | 15m |
| `mtf_hma_rsi_atr_dynamic_tp_v1.py` | 11.5 | -3.3% | 15m |
| `mtf_hma_supertrend_adx_kama_rsi_zscore_vol_15m_v1.py` | 10.9 | -3.8% | 15m |
| `regime_ensemble_hma_st_rsi_bbw_1h_4h_v1.py` | 10.0 | -7.0% | 1h |
| `mtf_hma_supertrend_rsi_adx_1h_4h_v1.py` | 9.6 | -5.2% | 1h |
| `mtf_hma_supertrend_rsi_bbw_vol_15m_4h_v1.py` | 9.0 | -4.7% | 15m |
| `mtf_hma_supertrend_adx_rsi_zscore_15m_v1.py` | 9.0 | -4.4% | 15m |
| `mtf_donchian_rsi_volume_atr_v1.py` | 6.7 | -4.3% | 1h |

Read the `timeframe` variable at the top of each file — it tells you which OHLCV data to load.

---

## Step 2: Load Price Data

**OHLCV Parquet files:** `data/processed/klines/{SYMBOL}/{TIMEFRAME}.parquet`

Symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
Timeframes: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`

```python
import pandas as pd

df = pd.read_parquet("data/processed/klines/BTCUSDT/1h.parquet")
```

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `open_time` | datetime64[ms, UTC] | Bar open timestamp |
| `open` | float64 | Open price (USDT) |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Close price |
| `volume` | float64 | Base asset volume (e.g., BTC) |
| `taker_buy_volume` | float64 | Taker buy volume |
| `trades` | int64 | Number of trades in bar |

**Train/test split:**
```python
train = df[df["open_time"] < "2025-01-01"].reset_index(drop=True)
test  = df[df["open_time"] >= "2025-01-01"].reset_index(drop=True)
```

**Funding rates** (optional, for advanced verification):
`data/processed/funding/{SYMBOL}/funding_rate.parquet`
Columns: `calc_time`, `funding_interval_hours` (8), `last_funding_rate`

---

## Step 3: Run the Strategy

Each strategy file exposes `generate_signals(prices)` which takes the DataFrame and returns a numpy array.

```python
import importlib.util
import numpy as np

# Load strategy
spec = importlib.util.spec_from_file_location("strat", "strategies/mtf_hma_rsi_atr_dynamic_tp_v1.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print(f"Strategy: {mod.name}, Timeframe: {mod.timeframe}, Leverage: {mod.leverage}")

# Load correct timeframe data
prices = pd.read_parquet(f"data/processed/klines/BTCUSDT/{mod.timeframe}.parquet")
train = prices[prices["open_time"] < "2025-01-01"].reset_index(drop=True)

# Generate signals
signals = mod.generate_signals(train)
# signals is numpy array, same length as train
# Values: positive = long, negative = short, 0 = flat
# The absolute value IS the position size (e.g., 0.35 = 35% of capital)
```

---

## Step 4: Write Your Own Backtest

**Rules the original engine claims to follow (verify these yourself):**

1. Signal at bar `t` → position changes at bar `t+1` open price (1-bar delay)
2. Fee: 0.04% taker per side + 0.01% slippage per side = 0.05% per side
3. Fees charged on BOTH entry AND exit
4. Position size = signal value × equity (e.g., signal=0.35 means 35% of current equity)
5. Starting capital: $10,000
6. Funding rate: applied every 8h on open positions (can skip for initial check)

```python
def my_backtest(prices, signals, leverage=1.0, initial_capital=10000.0):
    """
    Write your own backtest. Do NOT copy the project's backtest.py.
    """
    n = len(prices)
    cost_per_side = (0.04 + 0.01) / 100  # 0.05%

    # CRITICAL: 1-bar delay — signal[t] fills at bar[t+1] open
    delayed = np.zeros(n)
    delayed[1:] = signals[:-1]

    equity = initial_capital
    position = 0.0  # current position size as fraction
    trades = []

    for i in range(1, n):
        bar_open = prices["open"].iloc[i]
        bar_close = prices["close"].iloc[i]
        target = delayed[i]

        # Cost on position change
        change = target - position
        if abs(change) > 1e-8:
            fee = abs(change) * cost_per_side * leverage * equity
            equity -= fee

        # PnL on held position (from open to close of this bar)
        if abs(position) > 1e-8:
            ret = (bar_close - bar_open) / bar_open
            equity += equity * position * ret * leverage

        # Track trades (position flips)
        if abs(change) > 1e-8 and position * target <= 0 and abs(target) > 1e-8:
            trades.append({
                "time": str(prices["open_time"].iloc[i]),
                "direction": "LONG" if target > 0 else "SHORT",
                "price": bar_open,
                "size": abs(target),
            })

        position = target

        if equity <= 0:
            break

    total_return = (equity / initial_capital - 1) * 100
    returns = np.diff(np.log(np.maximum(1, [initial_capital] + [equity])))  # simplified

    return {
        "final_equity": equity,
        "return_pct": total_return,
        "num_trades": len(trades),
        "trades": trades,
    }

# Run it
result = my_backtest(train, signals, leverage=mod.leverage)
print(f"Return: {result['return_pct']:+.1f}%  Trades: {result['num_trades']}")
```

---

## Step 5: What to Check and Report

### Must verify:
1. **Does `generate_signals()` produce valid output?** — Same length as prices, values in [-1, 1]
2. **No look-ahead?** — Signal at bar `i` should NOT change if you add more data after `i`:
   ```python
   sig_1000 = mod.generate_signals(train.iloc[:1000])
   sig_full = mod.generate_signals(train)
   assert abs(sig_1000[-1] - sig_full[999]) < 1e-8, "LOOK-AHEAD DETECTED!"
   ```
3. **Does your backtest return match the claimed return?** — Should be within ~5%
4. **Drawdown** — Track peak equity, calculate max dropdown from peak
5. **Works on all 3 symbols?** — Run on BTCUSDT, ETHUSDT, SOLUSDT separately

### Red flags:
- Return > 100,000% — extreme compounding, verify trade-by-trade
- Sharpe > 10 — unusually high, check trade distribution over time
- Most trades in one year only — may be overfitting to bull/bear market
- Strategy uses `prices["open"]` in signal logic — potential subtle look-ahead since fills happen at open

### Report format:
```
Strategy: [name]
Symbol: [BTCUSDT/ETHUSDT/SOLUSDT]
Period: [train/test]
Your Return: [X%]  vs Claimed: [Y%]  Match: [yes/no]
Look-ahead test: [pass/fail]
Trades: [N]
Concerns: [any issues found]
```

---

## Environment

```bash
pip install pandas numpy pyarrow
# That's it. No API keys, no LLM, no special packages needed.
```

All data is already downloaded in `data/processed/`. Total ~571MB.

*Last updated: 2026-03-21*
