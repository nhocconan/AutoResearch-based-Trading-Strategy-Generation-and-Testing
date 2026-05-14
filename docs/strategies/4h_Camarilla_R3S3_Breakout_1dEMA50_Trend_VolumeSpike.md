# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.025 | +20.4% | -8.6% | 217 | DISCARD |
| ETHUSDT | 0.046 | +22.4% | -7.5% | 200 | KEEP |
| SOLUSDT | 0.692 | +68.5% | -10.1% | 171 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.530 | +11.5% | -7.0% | 71 | KEEP |
| SOLUSDT | 0.503 | +10.8% | -5.8% | 57 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA50 Trend Filter and Volume Spike (Revised)
- Uses Camarilla R3/S3 levels from 1d for breakout signals (stronger than Donchian)
- 1d EMA50 defines long-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.8x 20-period average) filters weak breakouts
- Exit when price crosses Camarilla midpoint OR crosses 1d EMA50
- Designed for 4h timeframe targeting 20-30 trades/year (80-120 over 4 years)
- Works in both bull and bear markets by following the 1d EMA50 trend filter
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_R3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need 1d EMA50, 1d Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND above 1d EMA50 AND volume spike
            if (close[i] > camarilla_R3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND below 1d EMA50 AND volume spike
            elif (close[i] < camarilla_S3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Camarilla midpoint OR crosses 1d EMA50
            exit_signal = False
            camarilla_mid = (camarilla_R3_aligned[i] + camarilla_S3_aligned[i]) / 2
            
            if position == 1:
                # Exit long when price < Camarilla midpoint OR < 1d EMA50
                if close[i] < camarilla_mid or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Camarilla midpoint OR > 1d EMA50
                if close[i] > camarilla_mid or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 16:30
