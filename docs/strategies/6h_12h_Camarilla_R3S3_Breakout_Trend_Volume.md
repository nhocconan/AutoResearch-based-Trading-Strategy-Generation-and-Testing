# Strategy: 6h_12h_Camarilla_R3S3_Breakout_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.044 | +21.7% | -11.0% | 205 | PASS |
| ETHUSDT | 0.040 | +20.6% | -15.7% | 194 | PASS |
| SOLUSDT | 0.926 | +144.9% | -16.6% | 187 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.008 | -4.5% | -7.8% | 67 | FAIL |
| ETHUSDT | 0.519 | +14.5% | -10.9% | 63 | PASS |
| SOLUSDT | 0.217 | +8.9% | -12.1% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "6h_12h_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_12h = close_12h > ema50_12h
    
    # Get 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    R3 = np.zeros(len(high_1d))
    S3 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        if i < 1:
            R3[i] = np.nan
            S3[i] = np.nan
        else:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Align indicators to 6h timeframe
    R3_12h_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h_aligned = align_htf_to_ltf(prices, df_1d, S3)
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_12h_aligned[i]) or 
            np.isnan(S3_12h_aligned[i]) or
            np.isnan(trend_up_12h_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume confirmation
            if (close[i] > R3_12h_aligned[i] and 
                trend_up_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < S3_12h_aligned[i] and 
                  not trend_up_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if (close[i] < S3_12h_aligned[i] or not trend_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if (close[i] > R3_12h_aligned[i] or trend_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 16:56
