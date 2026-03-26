# Strategy Code Rules (MUST READ BEFORE WRITING strategy.py)

## Rule 1: MTF Data Loading — ONCE Before Loop

```python
# CORRECT ✅
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    # Load HTF data ONCE, BEFORE the loop
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)

    signals = np.zeros(len(prices))
    for i in range(100, len(prices)):
        trend = hma_4h_aligned[i]  # ← use pre-aligned array, NOT get_htf_data
        ...

# WRONG ❌ — DO NOT DO THIS
def generate_signals(prices):
    signals = np.zeros(len(prices))
    for i in range(100, len(prices)):
        df_4h = get_htf_data(prices, '4h')  # ← LOADS FILE 45K TIMES = HANG
        idx = i // 16                        # ← MANUAL MTF = LOOK-AHEAD
```

**Why**: `get_htf_data()` reads a Parquet file from disk. Calling it 45,000 times
inside a for-loop takes hours. Call it ONCE and use the aligned array.

## Rule 2: No Manual MTF Index Mapping

```python
# WRONG ❌
idx_4h = i // 16  # uses unclosed 4h bar = look-ahead!
price_4h = close_4h[idx_4h]

# CORRECT ✅
# align_htf_to_ltf() handles this properly with shift(1)
trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_values)
# Inside loop:
trend = trend_4h_aligned[i]  # already aligned + shifted
```

**Why**: At 15m bar index 5 (01:15 UTC), `i // 16 = 0` gives the 4h bar at 00:00
which is STILL FORMING until 04:00. This is look-ahead. `align_htf_to_ltf` uses
`shift(1)` to only use the PREVIOUS completed 4h bar.

## Rule 3: No Manual Resampling

```python
# WRONG ❌
df.index = pd.date_range(start='2021-01-01', periods=n, freq='15min')
df_4h = df.resample('4h').agg(...)

# WRONG ❌
prices_idx = prices.set_index('open_time')
df_4h = prices_idx.resample('4h').agg(...)

# CORRECT ✅
from mtf_data import get_htf_data
df_4h = get_htf_data(prices, '4h')  # loads actual Binance 4h Parquet
```

**Why**: Resampling creates fake timestamps, misaligns on SOLUSDT (data gaps),
and doesn't match Binance's actual 4h bar boundaries.

## Rule 4: Position Sizing

```python
# Signal value = position size as fraction of capital
SIZE = 0.30  # 30% of capital

# CORRECT ✅ — discrete levels
signals[i] = SIZE      # long 30%
signals[i] = -SIZE     # short 30%
signals[i] = 0.0       # flat
signals[i] = SIZE / 2  # half position (take profit)

# WRONG ❌
signals[i] = 1.0   # 100% of capital = will blow up on 50% crash
signals[i] = 0.73  # non-discrete = fee churn on every tiny change
```

- MAX magnitude: 0.40
- Normal range: 0.20 to 0.35
- Use DISCRETE levels (0.0, ±0.15, ±0.30) to minimize fee churn
- Each signal change costs 0.10% round-trip fees

## Rule 5: No Look-Ahead

```python
# WRONG ❌
prices['close'].shift(-1)  # future data
prices.iloc[i + 1]         # future bar

# CORRECT ✅
# At bar i, only use prices.iloc[:i+1]
# All rolling/EMA/SMA with min_periods parameter
ema = close_s.ewm(span=21, min_periods=21, adjust=False).mean()
```

## Rule 6: Stoploss via Signal

```python
# Track entry price and set signal=0 when stopped out
if position_side == 1 and close[i] < entry_price - 2.0 * atr[i]:
    signals[i] = 0.0  # stoploss hit

if position_side == -1 and close[i] > entry_price + 2.0 * atr[i]:
    signals[i] = 0.0  # stoploss hit
```

## Rule 7: Available Data

Primary timeframes: `5m, 15m, 30m, 1h, 4h, 6h, 12h, 1d`
HTF reference: all above + `1w`

```python
# Strategy must declare:
name = "descriptive_name_v1"
timeframe = "1h"  # primary timeframe
leverage = 1.0    # always 1.0

# Prices DataFrame columns:
# open_time, open, high, low, close, volume, taker_buy_volume, trades
```

## Rule 8: Performance

- generate_signals() must complete in < 30 seconds for 45K bars
- Use vectorized numpy/pandas where possible
- For-loops OK for signal logic, but pre-compute indicators before the loop
- Never call I/O (file read, network) inside the loop

## Rule 9: MUST Generate Trades

A strategy that generates 0 trades is WORTHLESS and will be auto-rejected.
This has been the #1 repeated mistake. Your strategy MUST:
- Generate at least 10 trades on EACH symbol during train (2021-2024, 4 years)
- Generate at least 3 trades on EACH symbol during test (2025-2026, 15 months)
- If your entry conditions are too strict, LOOSEN them
- Test mentally: "would this trigger on a 20% BTC rally? On a 50% crash?"

Common causes of 0 trades:
- RSI threshold too narrow (e.g., only enter when RSI exactly 42-43)
- Multiple conflicting filters that never all agree
- ADX threshold too high (ADX > 40 rarely happens)
- Entry requires conditions that are mutually exclusive

## Summary Checklist

Before submitting strategy.py, verify:
- [ ] `get_htf_data()` called ONCE before loop, not inside
- [ ] No `i // N` manual MTF mapping
- [ ] No `.resample()` or `pd.date_range()`
- [ ] Signal values discrete: 0.0, ±0.15, ±0.30 (max 0.40)
- [ ] Stoploss logic present (signal → 0)
- [ ] leverage = 1.0
- [ ] **MUST generate ≥10 trades per symbol** (entry conditions not too strict)
- [ ] **ALL symbols must have Sharpe > 0** individually (no SOL-only strategies)
- [ ] All indicators use min_periods
- [ ] No .shift(-n) or future indexing

## Rule 10: Session/Hour Filters — Correct Pattern

`open_time` in the prices DataFrame is **already `datetime64[ms]`** (NOT milliseconds integer).
Using `pd.to_datetime(open_time, unit='ms')` will CRASH with TypeError.

```python
# WRONG ❌ — crashes: "floor_divide datetime64" TypeError
open_time = prices["open_time"].values
hours = pd.to_datetime(open_time, unit='ms').hour.values  # CRASH!

# WRONG ❌ — crashes: cannot floor_divide datetime64 scalar
timestamp_ms = open_time[i]
hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24  # CRASH!

# WRONG ❌ — also crashes
hour = (open_time[i] // 3600000) % 24  # CRASH!

# CORRECT ✅ — compute ONCE before loop, index inside
hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
for i in range(warmup, n):
    hour = hours[i]
    in_session = (8 <= hour <= 20)  # UTC 8-20

# CORRECT ✅ — alternative using open_time column (already datetime64, NOT ms int)
hours = pd.DatetimeIndex(prices["open_time"]).hour  # pre-compute before loop

# CORRECT ✅ — per-bar if you need one-off Timestamp
hour = pd.Timestamp(open_time[i]).hour  # safe but slower
```

**Why**: Binance parquet stores `open_time` as `datetime64[ms]` (NOT as integer milliseconds).
Every arithmetic operation on datetime64 (`//`, `/`, `*`) crashes with TypeError.
Use `prices.index.hour` — it's already a DatetimeIndex with no conversion needed.

## Rule 11: Trade Frequency by Timeframe

The #1 reason lower TF strategies fail: too many trades → fee drag.

| Primary TF | Max Trades/Year | Fee Drag at 0.05% RT |
|-----------|----------------|---------------------|
| 15m | Target: 50-100 | 2.5-5% |
| 30m | Target: 40-80 | 2-4% |
| 1h | Target: 30-60 | 1.5-3% |
| 4h | Target: 20-50 | 1-2.5% |
| 12h/1d | Target: 10-30 | 0.5-1.5% |

If your strategy generates >100 trades/year on any TF below 4h, your entry
conditions are TOO LOOSE. Add more filters (session, volume, confluence).

For 15m/30m/1h: use HTF (4h/12h) for signal direction, lower TF only for
entry timing precision. This gives HTF trade frequency with lower TF execution.
