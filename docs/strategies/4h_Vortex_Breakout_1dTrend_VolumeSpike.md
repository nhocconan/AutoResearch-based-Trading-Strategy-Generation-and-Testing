# Strategy: 4h_Vortex_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.271 | +14.4% | -6.1% | 75 | FAIL |
| ETHUSDT | 0.173 | +27.2% | -8.6% | 91 | PASS |
| SOLUSDT | 0.432 | +42.7% | -9.0% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.187 | +20.6% | -5.5% | 25 | PASS |
| SOLUSDT | -1.959 | -8.1% | -9.4% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_Vortex_Breakout_1dTrend_VolumeSpike
# Hypothesis: Vortex indicator (VI+ > VI-) identifies trend direction with less whipsaw than traditional methods.
# Enter long when VI+ crosses above VI- with volume spike and 1d EMA50 uptrend.
# Enter short when VI- crosses above VI+ with volume spike and 1d EMA50 downtrend.
# Exit when Vortex signal reverses.
# Uses 4h timeframe with 1d trend filter to balance trade frequency and win rate.
# Designed to work in both bull (buy in uptrend) and bear (sell in downtrend).
# Target: 20-40 trades/year per symbol.

name = "4h_Vortex_Breakout_1dTrend_VolumeSpike"
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

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Vortex Indicator (VI) on 4h data
    # True Range
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr[0] = tr0[0]  # First value

    # Positive and Negative Vortex Movements
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0

    # Sum over 14 periods
    tr14 = np.zeros(n)
    vm_plus14 = np.zeros(n)
    vm_minus14 = np.zeros(n)
    for i in range(14, n):
        tr14[i] = np.sum(tr[i-14:i])
        vm_plus14[i] = np.sum(vm_plus[i-14:i])
        vm_minus14[i] = np.sum(vm_minus[i-14:i])

    # VI+ and VI-
    vi_plus = np.zeros(n)
    vi_minus = np.zeros(n)
    for i in range(14, n):
        if tr14[i] != 0:
            vi_plus[i] = vm_plus14[i] / tr14[i]
            vi_minus[i] = vm_minus14[i] / tr14[i]

    # Vortex crossover signals
    vi_plus_cross_above = np.zeros(n, dtype=bool)
    vi_minus_cross_above = np.zeros(n, dtype=bool)
    for i in range(15, n):
        vi_plus_cross_above[i] = (vi_plus[i-1] <= vi_minus[i-1]) and (vi_plus[i] > vi_minus[i])
        vi_minus_cross_above[i] = (vi_minus[i-1] <= vi_plus[i-1]) and (vi_minus[i] > vi_plus[i])

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ crosses above VI- with volume spike and 1d EMA uptrend
            if vi_plus_cross_above[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ with volume spike and 1d EMA downtrend
            elif vi_minus_cross_above[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend reversal)
            if vi_minus_cross_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend reversal)
            if vi_plus_cross_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 05:23
