# Strategy: 6h_camarilla_pivot_1d_ema_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.305 | +28.5% | -2.6% | 351 | PASS |
| ETHUSDT | -0.403 | +11.8% | -6.1% | 350 | FAIL |
| SOLUSDT | 0.060 | +22.7% | -14.7% | 312 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.701 | -6.7% | -7.4% | 118 | FAIL |
| SOLUSDT | 0.718 | +11.5% | -5.9% | 107 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d EMA Trend and Volume Confirmation.
Long when price breaks above R4 with 1d uptrend and volume confirmation.
Short when price breaks below S4 with 1d downtrend and volume confirmation.
Exit when price crosses back below R3 (long) or above S3 (short).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D EMA TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === CAMARILLA PIVOT LEVELS (6H) ===
    # Based on previous day's OHLC
    cam_high = np.full(n, np.nan)
    cam_low = np.full(n, np.nan)
    cam_close = np.full(n, np.nan)
    
    # Calculate previous day's OHLC for each 6h bar
    for i in range(n):
        if i >= 4:  # Need at least 4 previous 6h bars for 1 day (24h/6h=4)
            prev_day_high = np.max(high[i-4:i])
            prev_day_low = np.min(low[i-4:i])
            prev_day_close = close[i-1]  # Previous bar's close as approximation
            cam_high[i] = prev_day_high
            cam_low[i] = prev_day_low
            cam_close[i] = prev_day_close
    
    # Camarilla levels
    cam_range = cam_high - cam_low
    r3 = cam_close + cam_range * 1.1 / 4
    r4 = cam_close + cam_range * 1.1 / 2
    s3 = cam_close - cam_range * 1.1 / 4
    s4 = cam_close - cam_range * 1.1 / 2
    
    # === VOLUME CONFIRMATION (6H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(one_d_ema_aligned[i]) or np.isnan(r3[i]) or np.isnan(r4[i]) or 
            np.isnan(s3[i]) or np.isnan(s4[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > one_d_ema_aligned[i]
        downtrend = close[i] < one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 OR trend turns down
            if close[i] < r3[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 OR trend turns up
            if close[i] > s3[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Camarilla breakout with trend alignment
            if close[i] > r4[i] and uptrend:
                # Breakout above R4 in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < s4[i] and downtrend:
                # Breakdown below S4 in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 21:50
