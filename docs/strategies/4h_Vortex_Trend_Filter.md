# Strategy: 4h_Vortex_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.847 | +81.9% | -8.6% | 584 | PASS |
| ETHUSDT | 1.097 | +134.3% | -11.2% | 577 | PASS |
| SOLUSDT | 1.381 | +333.4% | -21.6% | 583 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.274 | +2.4% | -8.2% | 207 | FAIL |
| ETHUSDT | 0.913 | +24.3% | -8.0% | 199 | PASS |
| SOLUSDT | 1.151 | +32.2% | -9.2% | 192 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Vortex_Trend_Filter
# Hypothesis: Use 1-day Vortex Indicator (VI+) and VI- to determine trend direction on 4h timeframe.
# Enter long when VI+ > VI- and price is above 4h EMA20; short when VI- > VI+ and price below 4h EMA20.
# Exit when trend reverses. Works in both bull and bear markets by following the dominant daily trend.
# Uses vortex to filter noise and EMA20 for dynamic support/resistance. Targets 20-40 trades/year.

name = "4h_Vortex_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Vortex calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex Indicator components (VM+ and VM-)
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum over 14 periods
    n_1d = len(high_1d)
    vi_plus = np.zeros(n_1d)
    vi_minus = np.zeros(n_1d)
    
    for i in range(14, n_1d):
        if i >= 14:
            sum_vm_plus = np.sum(vm_plus[i-13:i+1])
            sum_vm_minus = np.sum(vm_minus[i-13:i+1])
            sum_tr = np.sum(tr[i-13:i+1])
            if sum_tr > 0:
                vi_plus[i] = sum_vm_plus / sum_tr
                vi_minus[i] = sum_vm_minus / sum_tr
    
    # Align Vortex indicators to 4h timeframe
    vi_plus_4h = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_4h = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 4h EMA20 for dynamic support/resistance
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(vi_plus_4h[i]) or np.isnan(vi_minus_4h[i]) or 
            np.isnan(ema_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- and price above EMA20
            if vi_plus_4h[i] > vi_minus_4h[i] and close[i] > ema_20_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ and price below EMA20
            elif vi_minus_4h[i] > vi_plus_4h[i] and close[i] < ema_20_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend turns bearish (VI- > VI+) or price breaks below EMA20
            if vi_minus_4h[i] > vi_plus_4h[i] or close[i] < ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend turns bullish (VI+ > VI-) or price breaks above EMA20
            if vi_plus_4h[i] > vi_minus_4h[i] or close[i] > ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 00:52
