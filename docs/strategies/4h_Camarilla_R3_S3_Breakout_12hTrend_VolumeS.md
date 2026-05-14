# Strategy: 4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.285 | +32.0% | -7.4% | 240 | PASS |
| ETHUSDT | 0.131 | +26.1% | -15.2% | 210 | PASS |
| SOLUSDT | 0.367 | +46.5% | -21.5% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.225 | -3.5% | -7.4% | 90 | FAIL |
| ETHUSDT | 0.179 | +8.0% | -10.2% | 80 | PASS |
| SOLUSDT | -0.224 | +2.7% | -13.8% | 66 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS
Hypothesis: Uses Camarilla R3/S3 levels (1.0x range) with 12h EMA50 trend filter and volume spike (2x 48-bar avg) to capture high-probability breakouts on 4h timeframe. Designed for low trade frequency (19-50/year) to minimize fee drag while capturing strong directional moves. Works in both bull and bear by following 12h trend direction. Targets 75-200 total trades over 4 years.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Camarilla pivots (R3/S3: 1.0x range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily typical price and range
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    R3 = typical_price + (range_ * 1.0 / 4)
    S3 = typical_price - (range_ * 1.0 / 4)
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    
    # Volume confirmation: >2x 48-period MA (8 days of 4h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_48[i])
        
        # Breakout conditions at R3/S3
        long_breakout = close[i] > R3_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S3_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint of R3/S3
        midpoint = (R3_aligned[i] + S3_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 03:37
