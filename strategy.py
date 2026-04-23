#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
- Uses 4h Camarilla pivot levels (R1, S1) derived from previous 1d OHLC for breakout signals
- Long breakout: price > R1 + volume > 1.5x 20-period avg + price > 1d EMA50 (uptrend)
- Short breakdown: price < S1 + volume > 1.5x 20-period avg + price < 1d EMA50 (downtrend)
- Exit: price reverts to Camarilla pivot point (PP)
- 1d EMA50 provides stronger trend filter than EMA34 to reduce whipsaw
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 20-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
- R1/S1 levels are tighter breakout points than R3/S3, increasing signal quality
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla formulas:
    # PP = (high + low + close) / 3
    # R1 = PP + (high - low) * 1.1 / 4
    # S1 = PP - (high - low) * 1.1 / 4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > R1 + volume spike + price > 1d EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < S1 + volume spike + price < 1d EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0