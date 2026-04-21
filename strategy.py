#!/usr/bin/env python3
"""
1h_4d_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use 4-day (4h) and daily (1d) timeframes for directional bias via Camarilla levels, with 1h for precise entry. 
Long when price breaks above 4h R1 with volume > 1.5x average volume, short when breaks below 4h S1 with volume > 1.5x average volume. 
Exit when price crosses back through 4h pivot point. 
Session filter (08-20 UTC) reduces noise trades. Designed for 1h timeframe to target 60-150 total trades over 4 years.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns. Volume confirmation filters false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla pivot levels (based on previous 4h bar)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_4h = np.full_like(close_4h, np.nan)
    r1_4h = np.full_like(close_4h, np.nan)
    s1_4h = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(high_4h)):
        pp_4h[i] = (high_4h[i-1] + low_4h[i-1] + close_4h[i-1]) / 3.0
        r1_4h[i] = close_4h[i-1] + (high_4h[i-1] - low_4h[i-1]) * 1.1 / 12.0
        s1_4h[i] = close_4h[i-1] - (high_4h[i-1] - low_4h[i-1]) * 1.1 / 12.0
    
    # Shift to align with current 4h bar (levels are based on previous 4h bar)
    pp_4h = np.roll(pp_4h, 1)
    r1_4h = np.roll(r1_4h, 1)
    s1_4h = np.roll(s1_4h, 1)
    pp_4h[0] = np.nan
    r1_4h[0] = np.nan
    s1_4h[0] = np.nan
    
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Precompute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above 4h R1 + volume confirmation + session filter
            if price > r1_4h_aligned[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short conditions: break below 4h S1 + volume confirmation + session filter
            elif price < s1_4h_aligned[i] and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below 4h pivot point
            if price < pp_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above 4h pivot point
            if price > pp_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4d_Camarilla_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0