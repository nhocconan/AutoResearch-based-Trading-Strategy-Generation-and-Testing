# Strategy: 4h_Vortex_Trend_Confirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.315 | +37.1% | -9.4% | 307 | PASS |
| ETHUSDT | 0.245 | +34.4% | -12.1% | 300 | PASS |
| SOLUSDT | 0.930 | +148.7% | -15.1% | 254 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.884 | -3.2% | -9.1% | 102 | FAIL |
| ETHUSDT | 0.065 | +6.2% | -11.5% | 96 | PASS |
| SOLUSDT | 0.219 | +9.0% | -14.0% | 82 | PASS |

## Code
```python
#/usr/bin/env python3
# 4h_Vortex_Trend_Confirmation
# Hypothesis: Vortex Indicator identifies trend direction and strength. Combined with volume confirmation
# and daily EMA trend filter, it provides high-probability entries in both bull and bear markets.
# The Vortex Indicator is effective in trending markets and avoids whipsaws during consolidation.
# Designed for 4h timeframe to target 20-50 trades per year, minimizing fee drag.

name = "4h_Vortex_Trend_Confirmation"
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

    # Volume confirmation: 1.5x 20-period SMA (moderate threshold to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after VI needs 14 bars
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
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
            # EXIT LONG: Trend weakness (VI- > VI+)
            if vi_minus[i] > vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakness (VI+ > VI-)
            if vi_plus[i] > vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 19:16
