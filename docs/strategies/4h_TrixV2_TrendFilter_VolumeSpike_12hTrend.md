# Strategy: 4h_TrixV2_TrendFilter_VolumeSpike_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.315 | +11.5% | -7.1% | 129 | FAIL |
| ETHUSDT | 0.750 | +56.7% | -6.6% | 99 | PASS |
| SOLUSDT | 0.606 | +67.5% | -21.6% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.090 | +6.8% | -7.8% | 44 | PASS |
| SOLUSDT | 0.203 | +8.3% | -8.5% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_TrixV2_TrendFilter_VolumeSpike_12hTrend
# Hypothesis: TRIX (12) with zero-lag filtering (signal line crossover) + volume spike + 12h EMA trend filter on 4h timeframe.
# TRIX identifies momentum extremes; zero-lag reduces lag while preserving signal integrity.
# Volume spike confirms institutional participation; 12h EMA ensures alignment with higher timeframe trend.
# Designed for 4-8 trades/year per symbol to minimize fee drag while capturing high-probability moves.
# Works in bull/bear: long when TRIX > signal and above 12h EMA; short when TRIX < signal and below 12h EMA.

name = "4h_TrixV2_TrendFilter_VolumeSpike_12hTrend"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')

    # TRIX (12-period) with signal line (9-period EMA of TRIX)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) * 100
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (ema3 / np.roll(ema3, 1) - 1) * 100  # Percentage change
    trix[0] = 0  # First value undefined due to roll

    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values

    # 12h EMA50 trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: current volume > 2.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix_signal[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above signal line, above 12h EMA50, with volume spike
            if (trix[i] > trix_signal[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal line, below 12h EMA50, with volume spike
            elif (trix[i] < trix_signal[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal line or price breaks below 12h EMA50
            if trix[i] < trix_signal[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal line or price breaks above 12h EMA50
            if trix[i] > trix_signal[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 04:43
