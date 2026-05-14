# Strategy: 4h_KAMA_Trend_With_12h_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.142 | +26.9% | -12.3% | 312 | PASS |
| ETHUSDT | 0.413 | +46.5% | -21.8% | 281 | PASS |
| SOLUSDT | 0.352 | +50.1% | -29.6% | 269 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.240 | +2.6% | -6.8% | 267 | FAIL |
| ETHUSDT | 0.481 | +11.2% | -6.7% | 78 | PASS |
| SOLUSDT | 0.197 | +8.2% | -10.5% | 103 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_12h_Volume_Filter
Hypothesis: KAMA adapts to market efficiency, providing a robust trend filter.
On 4h timeframe, price above/below KAMA(10) with 12h volume > 1.5x 20-period average
triggers entries. Volume confirmation from higher timeframe reduces false breakouts.
Designed for 20-50 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
"""

name = "4h_KAMA_Trend_With_12h_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    volume_12h = df_12h['volume'].values

    # Calculate KAMA(10) on 4h close
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    vol = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if np.sum(vol[i-9:i+1]) > 0:
            er[i] = change[i] / np.sum(vol[i-9:i+1])
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate 12h volume average
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        kama_val = kama[i]
        vol_avg_val = vol_avg_20_12h_aligned[i]

        if np.isnan(kama_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + 12h volume surge
            if close[i] > kama_val and volume_12h[i // 3] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + 12h volume surge
            elif close[i] < kama_val and volume_12h[i // 3] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA
            if close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA
            if close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 21:29
