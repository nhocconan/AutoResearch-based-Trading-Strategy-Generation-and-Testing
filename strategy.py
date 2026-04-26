#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 4h timeframe combined with 4h EMA50 trend filter to avoid counter-trend trades.
Volume spike confirms institutional interest. Session filter (08-20 UTC) reduces noise.
Designed for 60-150 total trades over 4 years (15-37/year) with discrete position sizing (0.0, ±0.20).
Works in both bull and bear markets by aligning with higher timeframe trend.
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
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    camarilla_range = (df_4h['high'].values - df_4h['low'].values) * 1.1 / 12
    camarilla_R1 = df_4h['close'].values + camarilla_range * 1
    camarilla_S1 = df_4h['close'].values - camarilla_range * 1
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Close breaks above Camarilla R1 + price > 4h EMA50 (uptrend) + volume spike
        if close[i] > camarilla_R1_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Camarilla S1 + price < 4h EMA50 (downtrend) + volume spike
        elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 4h EMA50 in opposite direction
        elif position == 1 and close[i] < ema_50_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_4h_aligned[i]:
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

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0