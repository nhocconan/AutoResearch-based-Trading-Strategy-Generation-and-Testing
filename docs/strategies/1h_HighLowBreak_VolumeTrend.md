# Strategy: 1h_HighLowBreak_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.543 | -1.9% | -10.9% | 524 | FAIL |
| ETHUSDT | 0.116 | +25.5% | -14.3% | 509 | PASS |
| SOLUSDT | 0.450 | +59.9% | -22.4% | 513 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.809 | +17.4% | -9.6% | 152 | PASS |
| SOLUSDT | 0.471 | +13.0% | -8.4% | 157 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
1h_HighLowBreak_VolumeTrend
Hypothesis: Price breaking above recent 20-period high or below low with volume confirmation 
and aligned 4h/1d trend captures momentum. Using 1h for entry timing with 4h/1d trend filters 
reduces false signals. Target: 20-50 trades/year per symbol.
"""

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
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA20 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 20-period high/low breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 21  # need 20 for rolling + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high with volume spike and uptrend on both 4h and 1d
            if (close[i] > high_20[i] and volume_spike[i] and 
                close[i] > ema20_4h_aligned[i] and close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low with volume spike and downtrend on both 4h and 1d
            elif (close[i] < low_20[i] and volume_spike[i] and 
                  close[i] < ema20_4h_aligned[i] and close[i] < ema20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below 20-period low or trend fails on either timeframe
            if (close[i] < low_20[i] or 
                close[i] < ema20_4h_aligned[i] or close[i] < ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns above 20-period high or trend fails on either timeframe
            if (close[i] > high_20[i] or 
                close[i] > ema20_4h_aligned[i] or close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HighLowBreak_VolumeTrend"
timeframe = "1h"
leverage = 1.0
```

## Last Updated
2026-04-27 17:34
