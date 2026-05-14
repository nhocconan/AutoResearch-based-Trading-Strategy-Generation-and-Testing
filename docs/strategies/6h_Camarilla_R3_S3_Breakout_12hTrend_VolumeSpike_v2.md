# Strategy: 6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.041 | +21.7% | -10.1% | 240 | PASS |
| ETHUSDT | 0.136 | +26.7% | -11.4% | 231 | PASS |
| SOLUSDT | 0.616 | +82.6% | -21.2% | 207 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.065 | -4.7% | -12.6% | 96 | FAIL |
| ETHUSDT | 0.893 | +20.8% | -7.3% | 74 | PASS |
| SOLUSDT | -0.028 | +4.6% | -11.6% | 73 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Using Camarilla R3/S3 levels (wider bands) with 12h EMA50 trend filter and volume confirmation (1.5x average) reduces false breakouts while maintaining edge in both bull and bear markets. R3/S3 breakouts capture stronger moves, and the 12h trend filter ensures alignment with higher timeframe momentum. Target: 15-30 trades/year per symbol.
"""

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "6h"
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
    
    # Volume spike: >1.5x 50-period average (balanced frequency)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 12h EMA50 uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 12h EMA50 downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR closes below 12h EMA50
            if (close[i] < camarilla_s3_aligned[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR closes above 12h EMA50
            if (close[i] > camarilla_r3_aligned[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 12:45
