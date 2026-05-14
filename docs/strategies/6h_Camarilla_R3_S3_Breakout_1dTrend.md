# Strategy: 6h_Camarilla_R3_S3_Breakout_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.188 | +15.2% | -8.9% | 184 | FAIL |
| ETHUSDT | 0.011 | +20.9% | -8.1% | 166 | PASS |
| SOLUSDT | 1.113 | +125.3% | -12.5% | 156 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.468 | +11.5% | -7.0% | 64 | PASS |
| SOLUSDT | -0.217 | +2.9% | -8.3% | 61 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "6h"
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
    
    # Get 1d data for daily trend and previous close
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d trend: EMA34 > EMA89 = uptrend, EMA34 < EMA89 = downtrend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_up_1d = ema34_1d > ema89_1d
    trend_down_1d = ema34_1d < ema89_1d
    
    # Align 1d trend
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Previous day's close for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels (using previous day's range)
    R3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    R1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
    S1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in 1d uptrend with volume surge
            if (close[i] > R3_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in 1d downtrend with volume surge
            elif (close[i] < S3_aligned[i] and 
                  trend_down_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or 1d trend changes
            if (close[i] < R1_aligned[i] or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S1 or 1d trend changes
            if (close[i] > S1_aligned[i] or not trend_down_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 15:35
