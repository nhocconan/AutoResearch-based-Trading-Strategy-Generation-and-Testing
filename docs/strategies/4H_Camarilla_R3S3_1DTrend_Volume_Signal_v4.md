# Strategy: 4H_Camarilla_R3S3_1DTrend_Volume_Signal_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.161 | +27.4% | -8.2% | 92 | PASS |
| ETHUSDT | 0.399 | +42.6% | -10.3% | 79 | PASS |
| SOLUSDT | 0.819 | +106.8% | -17.7% | 69 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.099 | -4.1% | -7.7% | 31 | FAIL |
| ETHUSDT | 1.575 | +33.1% | -6.4% | 25 | PASS |
| SOLUSDT | -0.292 | +1.2% | -11.3% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R3S3_1DTrend_Volume_Signal_v4
# Hypothesis: Further refine the proven Camarilla R3/S3 breakout strategy by increasing the volume confirmation threshold to 2.5x and adding a 100-period volume moving average to reduce false signals and lower trade frequency. Uses 4h timeframe with 1d HTF for levels and trend. Target: 15-30 trades/year per symbol to stay well under the 400 trade limit, focusing on high-probability breakouts in both bull and bear markets.

name = "4H_Camarilla_R3S3_1DTrend_Volume_Signal_v4"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from previous daily period's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    hl_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * hl_range / 2
    s3_1d = close_1d - 1.1 * hl_range / 2
    
    # Align all levels to 4h timeframe (use previous daily period's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate EMA34 for trend filter (daily)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.5x average volume (100-period for stability)
    vol_ma = pd.Series(volume).rolling(window=100, min_periods=100).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R3, price above daily EMA34 (uptrend), volume spike (>2.5x)
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3, price below daily EMA34 (downtrend), volume spike (>2.5x)
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below daily S3 (opposite level)
            if close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above daily R3 (opposite level)
            if close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 02:55
