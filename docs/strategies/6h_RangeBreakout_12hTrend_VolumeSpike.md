# Strategy: 6h_RangeBreakout_12hTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.040 | +15.6% | -15.9% | 78 | FAIL |
| ETHUSDT | 0.201 | +32.0% | -16.9% | 80 | PASS |
| SOLUSDT | 1.134 | +246.2% | -25.2% | 75 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.694 | +20.4% | -8.0% | 22 | PASS |
| SOLUSDT | 0.002 | +4.1% | -17.4% | 26 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_RangeBreakout_12hTrend_VolumeSpike
Hypothesis: On 6h, buy breakouts above 20-period high when 12h EMA50 is rising and volume >1.5x average; sell breakdowns below 20-period low when 12h EMA50 is falling and volume >1.5x average. Uses volatility-adjusted breakout thresholds to avoid false breakouts in low-volatility regimes. Designed for low trade frequency (<30/year) to minimize fee gap while capturing strong trends in both bull and bear markets.
"""

name = "6h_RangeBreakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # 20-period high/low for breakout levels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Volatility filter: avoid breakouts in low-volatility regimes
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    # Use 20-period average of ATR to normalize breakout threshold
    atr_avg_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(atr_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Calculate dynamic breakout threshold based on volatility
        # In low volatility, require larger breakout; in high volatility, smaller breakout
        vol_factor = atr_avg_20[i] / (high[i] - low[i] + 1e-10)  # normalize by current bar range
        breakout_threshold = 0.001 * vol_factor  # base 0.1% threshold adjusted by volatility

        if position == 0:
            # LONG: Price breaks above 20-period high + 12h uptrend + volume spike + volatility filter
            if (close[i] > high_max_20[i-1] * (1 + breakout_threshold) and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + 12h downtrend + volume spike + volatility filter
            elif (close[i] < low_min_20[i-1] * (1 - breakout_threshold) and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low OR trend turns down
            if close[i] < low_min_20[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high OR trend turns up
            if close[i] > high_max_20[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 22:47
