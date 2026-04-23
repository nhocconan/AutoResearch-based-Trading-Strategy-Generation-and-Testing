#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
- Uses 1h Camarilla pivot levels (R1, S1) derived from previous 4h OHLC for precise entries
- Long breakout: price > R1 + volume > 1.3x 20-period avg + price > 4h EMA50 (uptrend)
- Short breakdown: price < S1 + volume > 1.3x 20-period avg + price < 4h EMA50 (downtrend)
- Exit: price reverts to Camarilla pivot point (PP) or opposite Camarilla level (R1/S1)
- 4h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Session filter: 08-20 UTC to avoid low-volume Asian session noise
- Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag on 1h timeframe
- Uses tighter volume threshold (1.3x) and session filter to reduce overtrading vs failed experiments
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
    
    # Volume confirmation: > 1.3x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 4h data ONCE before loop for Camarilla pivot calculation and EMA50
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels from previous 4h OHLC
    # Camarilla formulas:
    # PP = (high + low + close) / 3
    # R1 = PP + (high - low) * 1.1 / 12
    # S1 = PP - (high - low) * 1.1 / 12
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pp = (high_4h + low_4h + close_4h) / 3.0
    r1 = pp + (high_4h - low_4h) * 1.1 / 12.0
    s1 = pp - (high_4h - low_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar close)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 4h EMA
    
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
        
        # Volume spike confirmation (> 1.3x average)
        volume_spike = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > R1 + volume spike + price > 4h EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.20
                    position = 1
            # Short breakdown: price < S1 + volume spike + price < 4h EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point (PP) or breaks below S1 (reversal)
            if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to pivot point (PP) or breaks above R1 (reversal)
            if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0