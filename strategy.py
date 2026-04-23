#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
- Uses 12h Camarilla pivot levels (R3/S3) for breakout signals
- Long breakout: price > R3 + volume > 1.5x 20-period avg + price > 1w EMA50 (uptrend)
- Short breakdown: price < S3 + volume > 1.5x 20-period avg + price < 1w EMA50 (downtrend)
- Exit: price reverts to Camarilla pivot point (PP)
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
- Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets
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
    
    # Load 12h data ONCE before loop for Camarilla calculations
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h typical price for pivot
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tp_12h = typical_price_12h.values
    
    # Camarilla pivot calculations
    pp = (high_12h + low_12h + close_12h) / 3.0
    r3 = pp + (high_12h - low_12h) * 1.1 / 4.0
    s3 = pp - (high_12h - low_12h) * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe (wait for 12h bar close)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > R3 + volume spike + price > 1w EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < S3 + volume spike + price < 1w EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla pivot point
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla pivot point
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0