#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume spike (>1.8x 20-period median). 
Uses 4h HTF for trend alignment (more responsive than 1d) and discrete sizing (0.20) to minimize fee drag. 
Session filter (08-20 UTC) reduces noise. Target: 60-150 trades over 4 years = 15-37/year. Works in bull/bear via 4h trend filter + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Volume confirmation: volume > 1.8x 20-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for HTF trend filter (EMA20) and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate Camarilla levels for 4h (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 20-period volume median, 20-period EMA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Check session filter
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above R1 + volume confirm + bullish 4h trend
        if close[i] > camarilla_r1_aligned[i] and volume_confirm[i] and close[i] > ema20_4h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below S1 + volume confirm + bearish 4h trend
        elif close[i] < camarilla_s1_aligned[i] and volume_confirm[i] and close[i] < ema20_4h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price touches S1, short exits when price touches R1
        elif position == 1 and close[i] <= camarilla_s1_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= camarilla_r1_aligned[i]:
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0