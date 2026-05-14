# Strategy: 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.510 | +45.3% | -8.9% | 181 | KEEP |
| ETHUSDT | 0.023 | +19.9% | -13.4% | 183 | KEEP |
| SOLUSDT | 0.840 | +115.6% | -16.0% | 157 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.115 | -5.0% | -7.3% | 74 | DISCARD |
| ETHUSDT | 0.831 | +19.9% | -12.5% | 60 | KEEP |
| SOLUSDT | 0.105 | +7.0% | -8.3% | 52 | KEEP |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v4
# Hypothesis: Uses Camarilla R3/S3 levels from 1d as breakout levels, filtered by 1d EMA34 trend and volume spikes.
# Adds a momentum filter (4h RSI) to reduce overtrading and improve signal quality.
# Designed for 4h to balance trade frequency and signal quality. Works in bull/bear via trend filter.
# Only long when above 1d EMA34 and RSI > 50, short when below 1d EMA34 and RSI < 50.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v4"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike on 4h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 4h RSI for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(volume_spike[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 + above 1d EMA34 + volume spike + RSI > 50
            if close[i] > camarilla_r3_4h[i] and close[i] > ema_34_1d_4h[i] and volume_spike[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + below 1d EMA34 + volume spike + RSI < 50
            elif close[i] < camarilla_s3_4h[i] and close[i] < ema_34_1d_4h[i] and volume_spike[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below Camarilla S3 or below 1d EMA34
            if close[i] < camarilla_s3_4h[i] or close[i] < ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above Camarilla R3 or above 1d EMA34
            if close[i] > camarilla_r3_4h[i] or close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 00:24
