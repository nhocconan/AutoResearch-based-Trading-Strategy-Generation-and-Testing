#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm
Hypothesis: Fade extreme Camarilla levels (R3/S3) with 1d trend filter and volume confirmation on 6h timeframe.
Works in ranging markets by fading reversals at extreme levels, and in trending markets by only taking fades
that align with the higher timeframe trend (counter-trend within trend). Uses discrete position sizing (0.0, ±0.25)
to minimize fee churn. Target: 50-150 total trades over 4 years.
"""

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
    
    # Load 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter (aligned to 6h)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla R3 = Close + (High - Low) * 1.1 / 4
    # Camarilla S3 = Close - (High - Low) * 1.1 / 4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    camarilla_R3 = close_1d + camarilla_range
    camarilla_S3 = close_1d - camarilla_range
    
    # Align Camarilla levels to 6h (1d values available after the 1d bar closes)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long fade: price < Camarilla S3 (extreme low) + price > 1d EMA34 (uptrend filter) + volume spike
        if close[i] < camarilla_S3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short fade: price > Camarilla R3 (extreme high) + price < 1d EMA34 (downtrend filter) + volume spike
        elif close[i] > camarilla_R3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to mean (crosses Camarilla pivot point)
        # Camarilla pivot point = (High + Low + Close) / 3
        elif position == 1 and close[i] > (high_1d[i//16] + low_1d[i//16] + close_1d[i//16]) / 3:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] < (high_1d[i//16] + low_1d[i//16] + close_1d[i//16]) / 3:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0