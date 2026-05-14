# Strategy: 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.226 | +29.6% | -6.5% | 89 | PASS |
| ETHUSDT | 0.156 | +27.6% | -9.4% | 78 | PASS |
| SOLUSDT | 0.150 | +27.5% | -24.3% | 77 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.972 | -2.7% | -6.8% | 37 | FAIL |
| ETHUSDT | 0.386 | +11.3% | -6.4% | 29 | PASS |
| SOLUSDT | -0.368 | +0.2% | -12.2% | 29 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeFilter
Hypothesis: Camarilla pivot R3/S3 breakouts with 1d EMA34 trend filter and volume spikes capture institutional breakout moves in both bull and bear markets. Camarilla levels identify key support/resistance where institutional order flow concentrates. The 1d EMA34 filter ensures trading with the daily trend, while volume confirmation avoids false breakouts. Targets 12-37 trades/year on 12h timeframe.
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
    
    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d candle
    # Using previous day's high, low, close to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: >1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions: break of Camarilla R3/S3 with volume
        breakout_R3 = close[i] > R3_aligned[i]
        breakdown_S3 = close[i] < S3_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_R3
        short_entry = vol_confirm and downtrend and breakdown_S3
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_S3 or (not uptrend)
        short_exit = breakout_R3 or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-28 03:14
