# Strategy: 6h_1dPivot_R1S1_Volume_12hEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.081 | +18.3% | -11.0% | 171 | DISCARD |
| ETHUSDT | 0.462 | +41.6% | -9.6% | 152 | KEEP |
| SOLUSDT | 0.639 | +69.7% | -14.9% | 119 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.307 | +23.0% | -5.5% | 57 | KEEP |
| SOLUSDT | 0.128 | +7.2% | -6.8% | 46 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-day pivot points (R1/S1) and volume confirmation, filtered by 12-hour EMA34 trend.
Uses mean reversion at daily pivot levels with breakout confirmation. Designed to work in both bull and bear markets
by trading reversals at key daily levels with volume filter to avoid false breakouts. Aims for 12-37 trades/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align 1d pivot levels to 6h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume, and above 12h EMA34
            if close[i] > r1_1d_aligned[i] and volume_filter[i] and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, and below 12h EMA34
            elif close[i] < s1_1d_aligned[i] and volume_filter[i] and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below pivot
            if close[i] < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot
            if close[i] > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dPivot_R1S1_Volume_12hEMA34"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 15:17
