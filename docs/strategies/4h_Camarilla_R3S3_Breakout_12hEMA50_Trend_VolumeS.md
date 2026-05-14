# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeS

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.269 | +33.1% | -10.0% | 200 | PASS |
| ETHUSDT | 0.117 | +25.5% | -14.7% | 196 | PASS |
| SOLUSDT | 0.774 | +107.1% | -24.7% | 164 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.020 | -4.5% | -7.4% | 76 | FAIL |
| ETHUSDT | 0.797 | +19.4% | -12.3% | 69 | PASS |
| SOLUSDT | 0.021 | +5.6% | -10.6% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Use Camarilla R3/S3 levels from daily pivot for breakout entries, confirmed by 12h EMA50 trend and volume spikes.
# This combines price channel breakout with trend alignment and volume confirmation for high-probability setups.
# Designed to work in both bull and bear markets by filtering counter-trend trades and avoiding overtrading.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate Camarilla levels from previous day's range
    # Use daily data to get prior day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    # Shift to use previous day's data (avoid look-ahead)
    phigh = np.roll(df_1d['high'].values, 1)
    plow = np.roll(df_1d['low'].values, 1)
    pclose = np.roll(df_1d['close'].values, 1)
    # First value will be invalid due to roll, but we'll handle via min_periods later

    # Camarilla calculations
    range_val = phigh - plow
    R3 = pclose + (range_val * 1.1 / 4)
    S3 = pclose - (range_val * 1.1 / 4)

    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R3 with volume spike and uptrend
            if close[i] > R3_aligned[i] and volume_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and downtrend
            elif close[i] < S3_aligned[i] and volume_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < S3_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > R3_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 04:21
