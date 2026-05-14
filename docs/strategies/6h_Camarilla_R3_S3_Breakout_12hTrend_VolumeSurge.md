# Strategy: 6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSurge

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.264 | +31.9% | -9.5% | 174 | PASS |
| ETHUSDT | 0.045 | +21.7% | -12.6% | 163 | PASS |
| SOLUSDT | 0.478 | +59.8% | -18.6% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.727 | -8.7% | -13.3% | 66 | FAIL |
| ETHUSDT | 1.259 | +25.9% | -7.2% | 53 | PASS |
| SOLUSDT | 0.151 | +7.7% | -8.0% | 45 | PASS |

## Code
```python
# [134519] 6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSurge
# Hypothesis: Breakout at 1d Camarilla R3/S3 with volume surge and 12h trend filter works in both bull and bear markets.
# In bull markets: 12h trend up, breakouts above R3 capture continuation.
# In bear markets: 12h trend down, breakdowns below S3 capture continuation.
# Volume surge confirms institutional participation. 6h timeframe reduces noise vs lower timeframes.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSurge"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Camarilla levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_prev_1d = df_1d['close'].values
    high_prev_1d = df_1d['high'].values
    low_prev_1d = df_1d['low'].values
    range_prev_1d = high_prev_1d - low_prev_1d
    # R3 and S3 levels
    r3 = close_prev_1d + range_prev_1d * 1.1 / 4
    s3 = close_prev_1d - range_prev_1d * 1.1 / 4
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume surge: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~18 hours (3*6h) to reduce trade frequency
    
    start_idx = max(20, 1)  # Ensure enough data for volume and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R3 with volume surge in 12h uptrend
            if (close[i] > r3_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S3 with volume surge in 12h downtrend
            elif (close[i] < s3_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below S3 or 12h trend changes to down
            if close[i] < s3_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or 12h trend changes to up
            if close[i] > r3_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout captures institutional breakout moves in both bull and bear markets.
# Long when price breaks above 1d Camarilla R3 with volume surge and 12h uptrend.
# Short when price breaks below 1d Camarilla S3 with volume surge and 12h downtrend.
# Works in bull markets (sustained uptrend with breakouts above R3) and bear markets (sustained downtrend with breakdowns below S3).
# Volume surge confirms institutional participation. 6h timeframe balances signal quality and trade frequency.
# Discrete position sizing (0.25) balances risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
```

## Last Updated
2026-05-07 13:13
