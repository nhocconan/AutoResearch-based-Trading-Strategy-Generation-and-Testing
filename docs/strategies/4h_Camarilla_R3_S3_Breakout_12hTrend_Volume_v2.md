# Strategy: 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.169 | +27.5% | -11.2% | 350 | PASS |
| ETHUSDT | 0.159 | +27.7% | -14.7% | 344 | PASS |
| SOLUSDT | 0.335 | +44.7% | -24.6% | 330 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.949 | -1.8% | -7.0% | 125 | FAIL |
| ETHUSDT | 0.566 | +13.4% | -9.7% | 121 | PASS |
| SOLUSDT | 1.196 | +23.9% | -8.9% | 110 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_v2
Hypothesis: Price breaks above R3 or below S3 daily Camarilla levels with 12-hour EMA trend confirmation and volume spike. Designed for both bull and bear markets by aligning with 12-hour trend. Targets 20-40 trades/year to minimize fee drift.
"""

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_v2"
timeframe = "4h"
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
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align Camarilla levels to 4h timeframe (they change only at daily boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 12-hour EMA trend filter (per requirement: HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 12h uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 12h downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below R3 or drops below 12h EMA
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above S3 or rises above 12h EMA
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 01:26
