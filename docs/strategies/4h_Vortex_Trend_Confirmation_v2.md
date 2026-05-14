# Strategy: 4h_Vortex_Trend_Confirmation_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.592 | +50.2% | -7.4% | 219 | PASS |
| ETHUSDT | 0.640 | +60.0% | -8.5% | 206 | PASS |
| SOLUSDT | 0.944 | +134.5% | -17.5% | 173 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.986 | -3.2% | -7.9% | 81 | FAIL |
| ETHUSDT | 0.443 | +12.5% | -9.6% | 75 | PASS |
| SOLUSDT | -0.006 | +5.2% | -7.8% | 53 | FAIL |

## Code
```python
#/usr/bin/env python3
# 4h_Vortex_Trend_Confirmation_v2
# Hypothesis: Vortex Indicator identifies trend direction and strength. Combined with volume confirmation
# and daily EMA trend filter, it provides high-probability entries in both bull and bear markets.
# Reduced trading frequency by increasing volume threshold and adding hysteresis to avoid whipsaws.
# Designed for 4h timeframe to target 25-40 trades per year, minimizing fee drag.

name = "4h_Vortex_Trend_Confirmation_v2"
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

    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate Vortex Indicator (VI) on 4h data
    # VI+ = |current high - prior low|, VI- = |current low - prior high|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Set first values to avoid roll issues
    vm_plus[0] = 0
    vm_minus[0] = 0
    tr[0] = tr1[0]
    
    # Smooth using 14-period sums (standard Vortex)
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI- (normalized)
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum

    # Get daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 2.0x 30-period SMA (higher threshold to reduce trades)
    volume_series = pd.Series(volume)
    volume_sma30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_threshold = volume_sma30 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after VI needs 14 bars
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (bullish trend) with volume confirmation and uptrend
            if (vi_plus[i] > vi_minus[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish trend) with volume confirmation and downtrend
            elif (vi_minus[i] > vi_plus[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakness (VI- > VI+) OR price below EMA
            if vi_minus[i] > vi_plus[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakness (VI+ > VI-) OR price above EMA
            if vi_plus[i] > vi_minus[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 19:20
