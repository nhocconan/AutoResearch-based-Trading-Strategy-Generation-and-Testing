#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 1-week EMA50 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 4h timeframe combined with 1-week EMA50 trend filter to avoid counter-trend trades.
Volume spike confirms institutional interest. Designed for 75-200 total trades over 4 years (19-50/year) with 
discrete position sizing (0.0, ±0.25). Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from previous 4h bar
    camarilla_range = (df_4h['high'].values - df_4h['low'].values) * 1.1 / 12
    camarilla_R3 = df_4h['close'].values + camarilla_range * 3
    camarilla_S3 = df_4h['close'].values - camarilla_range * 3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Close breaks above Camarilla R3 + price > 1w EMA50 (uptrend) + volume spike
        if close[i] > camarilla_R3_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Camarilla S3 + price < 1w EMA50 (downtrend) + volume spike
        elif close[i] < camarilla_S3_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 1w EMA50 in opposite direction
        elif position == 1 and close[i] < ema_50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_1w_aligned[i]:
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

name = "4h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0