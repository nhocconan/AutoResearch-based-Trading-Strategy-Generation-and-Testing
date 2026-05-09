#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeS
Hypothesis: Breakouts from daily Camarilla R1/S1 levels with weekly EMA trend filter and volume spike confirmation.
The weekly timeframe provides a strong trend filter that works in both bull and bear markets.
Volume spike (>2x 24-period average) confirms breakout strength. Designed for low trade frequency (7-25/year)
to minimize fee drag on 1d timeframe.
"""

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeS"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Calculate daily Camarilla levels (R1, S1 are the key breakout levels)
    rang = ph - pl
    r1 = pc + 1.1 * rang * 1.0833  # R1 = Close + 1.1 * (High-Low) * 1.0833
    s1 = pc - 1.1 * rang * 1.0833  # S1 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align daily Camarilla levels to 1d timeframe (no alignment needed as same timeframe)
    r1_aligned = r1
    s1_aligned = s1
    
    # Volume spike filter: current volume / 24-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 20)  # Ensure volume MA and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R1 AND weekly uptrend (price > weekly EMA20) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND weekly downtrend (price < weekly EMA20) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below S1 OR trend reversal (price < weekly EMA20)
                if close[i] < s1_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above R1 OR trend reversal (price > weekly EMA20)
                if close[i] > r1_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals