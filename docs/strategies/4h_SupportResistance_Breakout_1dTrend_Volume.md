# Strategy: 4h_SupportResistance_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.467 | +41.9% | -8.3% | 176 | PASS |
| ETHUSDT | 0.019 | +20.1% | -12.3% | 168 | PASS |
| SOLUSDT | 0.925 | +126.3% | -20.4% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.569 | -8.4% | -9.1% | 68 | FAIL |
| ETHUSDT | 0.414 | +11.8% | -8.9% | 63 | PASS |
| SOLUSDT | -0.377 | +0.2% | -9.1% | 51 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_SupportResistance_Breakout_1dTrend_Volume
# Hypothesis: Buy when price breaks above daily resistance (previous day high) with 1d EMA uptrend and volume spike.
# Sell when price breaks below daily support (previous day low) with 1d EMA downtrend and volume spike.
# Exit when price returns to the daily midpoint (support + resistance)/2 to capture mean reversion.
# Uses daily levels to avoid overtrading and focuses on strong breakouts in trending markets.
# Target: 20-30 trades/year on 4h to minimize fee drag while capturing strong moves in both bull and bear markets.

name = "4h_SupportResistance_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily support/resistance: previous day's high and low
    resistance = np.roll(high_1d, 1)  # Previous day's high
    support = np.roll(low_1d, 1)      # Previous day's low
    midpoint = (resistance + support) / 2.0  # Exit level: average of support/resistance

    # Handle first day (no previous day)
    resistance[0] = np.nan
    support[0] = np.nan
    midpoint[0] = np.nan

    # Align daily levels to 4h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1d, support)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2.0x 20-period average (to filter weak moves)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above resistance + price > 1d EMA34 + volume spike
            if (close[i] > resistance_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below support + price < 1d EMA34 + volume spike
            elif (close[i] < support_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below midpoint (mean reversion)
            if close[i] < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above midpoint
            if close[i] > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 00:27
