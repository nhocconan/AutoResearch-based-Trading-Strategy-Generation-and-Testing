# Strategy: 4h_Keltner_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.212 | +29.4% | -10.2% | 104 | KEEP |
| ETHUSDT | 0.304 | +36.2% | -10.5% | 96 | KEEP |
| SOLUSDT | 0.743 | +91.7% | -16.8% | 91 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.605 | +1.0% | -5.6% | 38 | DISCARD |
| ETHUSDT | 0.631 | +15.3% | -7.6% | 36 | KEEP |
| SOLUSDT | 0.236 | +8.9% | -10.6% | 33 | KEEP |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_Volume_Trend
# Hypothesis: Use Keltner Channel breakout (ATR-based) for directional entries on 4h, confirmed by 1d EMA trend and volume spikes (>2x 20-period average). Enter long on upper band break with uptrend, short on lower band break with downtrend. Exit on middle band reversion. Targets 20-40 trades/year to minimize fee drag and work in both bull/bear via trend filter.

name = "4h_Keltner_Breakout_Volume_Trend"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate ATR(20) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Calculate EMA(20) for Keltner Channel middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channel bands: ATR multiplier = 2.0
    upper_band = ema20 + 2.0 * atr
    lower_band = ema20 - 2.0 * atr
    middle_band = ema20

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above upper band + price > 1d EMA50 + volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below lower band + price < 1d EMA50 + volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below middle band
            if close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above middle band
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 00:03
