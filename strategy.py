#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d SuperTrend + volume spike.
- Williams %R(14): mean reversion signal (oversold < -80, overbought > -20)
- 1d SuperTrend(10,3): HTF trend filter (only long in uptrend, short in downtrend)
- Volume spike: > 2.0x 20-period average for conviction
- Long: Williams %R crosses above -80 + volume spike + 1d SuperTrend uptrend
- Short: Williams %R crosses below -20 + volume spike + 1d SuperTrend downtrend
- Exit: Opposite Williams %R crossover (-20 for long, -80 for short) or SuperTrend flip
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h
- Discrete sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d SuperTrend(ATR=10, mult=3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # SuperTrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
        
        # Adjust bands
        if direction[i] == 1 and upper_band[i] < supertrend[i-1]:
            upper_band[i] = supertrend[i-1]
        if direction[i] == -1 and lower_band[i] > supertrend[i-1]:
            lower_band[i] = supertrend[i-1]
    
    # Align SuperTrend direction to 6h
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need 34 for ATR(10) warmup, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(supertrend_direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R crossovers (using previous bar to detect actual cross)
        prev_williams_r = williams_r[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 + volume spike + 1d SuperTrend uptrend
            if (williams_r[i] > -80 and prev_williams_r <= -80 and 
                volume_spike and 
                supertrend_direction_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 + volume spike + 1d SuperTrend downtrend
            elif (williams_r[i] < -20 and prev_williams_r >= -20 and 
                  volume_spike and 
                  supertrend_direction_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 OR SuperTrend flips down
            if (williams_r[i] > -20 and prev_williams_r <= -20) or supertrend_direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 OR SuperTrend flips up
            if (williams_r[i] < -80 and prev_williams_r >= -80) or supertrend_direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_SuperTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0