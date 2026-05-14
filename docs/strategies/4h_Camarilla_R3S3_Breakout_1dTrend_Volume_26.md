# Strategy: 4h_Camarilla_R3S3_Breakout_1dTrend_Volume_26

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.118 | +25.6% | -9.4% | 39 | PASS |
| ETHUSDT | 0.115 | +25.3% | -15.4% | 35 | PASS |
| SOLUSDT | 1.175 | +198.3% | -16.8% | 30 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.363 | +2.0% | -6.1% | 16 | FAIL |
| ETHUSDT | 0.305 | +10.4% | -8.8% | 12 | PASS |
| SOLUSDT | -1.084 | -13.4% | -23.8% | 14 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_26"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get daily data for Camarilla levels
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    high_low_range = high_1d - low_1d
    camarilla_high = high_1d + 1.1 * high_low_range
    camarilla_low = low_1d - 1.1 * high_low_range
    camarilla_range = camarilla_high - camarilla_low
    
    R3 = camarilla_low + camarilla_range * 1.1000
    S3 = camarilla_high - camarilla_range * 1.1000
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + above 1d EMA34 + volume
            if (close[i] > R3_4h[i] and
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + below 1d EMA34 + volume
            elif (close[i] < S3_4h[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price falls back below S3 or below 1d EMA34
            if (close[i] < S3_4h[i] or
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises back above R3 or above 1d EMA34
            if (close[i] > R3_4h[i] or
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 19:19
