#!/usr/bin/env python3
"""
1h_HTF_Camarilla_Pivot_1dTrend_VolumeConfirm_v1
Hypothesis: Use 1d Camarilla pivot R3/S3 breakouts for direction (from 4h/1d HTF), 
with 1d EMA34 trend filter and volume confirmation on 1h for precise entry. 
Only long when price > R3 and > 1d EMA34 with volume spike, short when price < S3 and < 1d EMA34 with volume spike.
Uses discrete position sizing (0.0, ±0.20) to minimize fee churn. Target: 60-150 trades over 4 years (15-37/year).
Works in both bull and bear markets by combining HTF structure (Camarilla pivots) with trend and volume filters.
"""

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
    
    # Calculate EMA13 for Elder Ray (primary timeframe) - not used, keeping for potential future use
    # ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Load 1d data for Camarilla pivot and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivots on 1d data
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4), 
    #            S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range / 4
    s3 = close_1d - camarilla_range / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Discrete position sizing
        base_size = 0.20
        
        # Long logic: price > R3 (breakout above resistance) + price > 1d EMA34 (trend up) + volume spike
        if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price < S3 (breakdown below support) + price < 1d EMA34 (trend down) + volume spike
        elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: loss of breakout condition or volume confirmation
        elif position == 1 and (close[i] <= r3_aligned[i] or close[i] <= ema_34_1d_aligned[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= s3_aligned[i] or close[i] >= ema_34_1d_aligned[i] or not volume_spike[i]):
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

name = "1h_HTF_Camarilla_Pivot_1dTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0