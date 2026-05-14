# Strategy: 12h_Pivot_Reversal_Strategy

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.318 | +9.0% | -18.8% | 195 | FAIL |
| ETHUSDT | 0.154 | +27.3% | -10.0% | 175 | PASS |
| SOLUSDT | 0.289 | +40.5% | -22.9% | 173 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.692 | +15.5% | -5.7% | 64 | PASS |
| SOLUSDT | -0.660 | -3.8% | -15.1% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_Pivot_Reversal_Strategy
# Hypothesis: Uses daily pivot points (PP, R1, S1) and 1-day trend to capture reversals at key levels.
# In bull markets, buy at S1 with bullish trend; in bear markets, sell at R1 with bearish trend.
# The pivot points act as support/resistance, and the 1-day trend filter ensures we trade with the higher timeframe momentum.
# Volume confirmation reduces false breakouts. Target: 15-25 trades/year per symbol to stay within trade limits.

timeframe = "12h"
name = "12h_Pivot_Reversal_Strategy"
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
    
    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pivot_point = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot_point - low_1d
    s1 = 2 * pivot_point - high_1d
    
    # Align pivot points to 12h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 1.5x average volume (24-period = 2 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S1 with volume, and 1d trend is bullish (close > EMA50)
            if (close[i] > s1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume, and 1d trend is bearish (close < EMA50)
            elif (close[i] < r1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot point (mean reversion to pivot)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above pivot point (mean reversion to pivot)
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 02:22
