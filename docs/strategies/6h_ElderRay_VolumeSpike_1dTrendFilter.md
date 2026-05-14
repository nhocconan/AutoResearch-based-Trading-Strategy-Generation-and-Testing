# Strategy: 6h_ElderRay_VolumeSpike_1dTrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.094 | +24.3% | -11.5% | 184 | KEEP |
| ETHUSDT | 0.062 | +21.8% | -16.0% | 166 | KEEP |
| SOLUSDT | 1.335 | +271.2% | -22.3% | 129 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.726 | -2.5% | -9.5% | 71 | DISCARD |
| ETHUSDT | 0.137 | +7.5% | -8.7% | 57 | KEEP |
| SOLUSDT | -0.336 | -1.4% | -15.9% | 55 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
6h Elder Ray Index + Volume Spike + 1d Trend Filter
Hypothesis: Elder Ray (bull/bear power) identifies institutional buying/selling pressure. Combined with volume spikes (confirming institutional participation) and 1d trend filter (avoiding counter-trend trades), it captures strong momentum moves while avoiding whipsaws. Works in both bull and bear markets by following the 1d trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    alpha = 2 / (period + 1)
    ema = np.zeros_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_elder_ray(high, low, close, ema_period=13):
    """Calculate Elder Ray Index: Bull Power and Bear Power"""
    ema = calculate_ema(close, ema_period)
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Elder Ray on 6h data
    bull_power, bear_power = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, 21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1 = uptrend (price > EMA), -1 = downtrend (price < EMA)
        trend = 1 if close[i] > ema_1d_aligned[i] else -1
        
        if position == 0:
            # Enter long: bull power > 0 (buying pressure) + volume spike + uptrend
            if (bull_power[i] > 0 and 
                vol_spike[i] and 
                trend == 1):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power < 0 (selling pressure) + volume spike + downtrend
            elif (bear_power[i] < 0 and 
                  vol_spike[i] and 
                  trend == -1):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative or trend changes
            if bull_power[i] <= 0 or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns positive or trend changes
            if bear_power[i] >= 0 or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_VolumeSpike_1dTrendFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 07:21
