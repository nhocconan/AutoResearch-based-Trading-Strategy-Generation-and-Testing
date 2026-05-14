# Strategy: 4H_Camarilla_R3S3_Breakout_1dWilliamsR_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.420 | +35.2% | -6.3% | 331 | PASS |
| ETHUSDT | 0.052 | +22.6% | -10.7% | 319 | PASS |
| SOLUSDT | -0.158 | +9.5% | -18.7% | 256 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.551 | -4.3% | -6.1% | 134 | FAIL |
| ETHUSDT | 0.645 | +13.7% | -6.1% | 122 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d Williams %R overbought/oversold filter and volume spike confirmation.
Uses Camarilla pivot levels from 4h timeframe for entry, combined with
1d Williams %R to avoid extreme overbought/oversold conditions that often reverse.
Volume spike confirms breakout momentum. Designed for 4h timeframe to reduce trade frequency.
Works in both bull and bear markets by avoiding counter-trend entries during exhaustion.
Target: 19-50 trades/year per symbol (75-200 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
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
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels for 4h timeframe
    range_4h = high_4h - low_4h
    camarilla_r3 = close_4h + (range_4h * 1.1 / 4)
    camarilla_s3 = close_4h - (range_4h * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (previous 4h bar values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 1d Williams %R (14-period) for overbought/oversold filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # need volume MA20 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Camarilla R3 AND NOT overbought (Williams %R > -80) AND volume spike
            if close[i] > camarilla_r3_aligned[i] and williams_r_aligned[i] > -80 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND NOT oversold (Williams %R < -20) AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and williams_r_aligned[i] < -20 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S3 for longs, R3 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S3
                if close[i] < camarilla_s3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dWilliamsR_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 15:26
