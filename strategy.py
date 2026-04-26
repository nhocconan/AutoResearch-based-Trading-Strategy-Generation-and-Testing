#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA34 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from daily timeframe combined with 1-week EMA34 trend filter to avoid counter-trend trades.
Volume spike confirms institutional interest. Designed for 30-100 total trades over 4 years (7-25/year) with 
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
    
    # Load 1d data ONCE before loop (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_range = (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    camarilla_R1 = df_1d['close'].values + camarilla_range * 1
    camarilla_S1 = df_1d['close'].values - camarilla_range * 1
    
    # Align Camarilla levels to 1d timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(50, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Close breaks above Camarilla R1 + price > 1w EMA34 (uptrend) + volume spike
        if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Camarilla S1 + price < 1w EMA34 (downtrend) + volume spike
        elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 1w EMA34 in opposite direction
        elif position == 1 and close[i] < ema_34_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_34_1w_aligned[i]:
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

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0