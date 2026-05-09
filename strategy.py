#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Breakouts from weekly Camarilla R1/S1 levels with 1w EMA trend filter and volume spike confirmation.
Weekly EMA provides robust trend filter that adapts to bull/bear markets. Volume spike (>2x 24-period average)
confirms breakout strength. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for Camarilla calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate weekly Camarilla levels (R1, S1 are the key breakout levels)
    rang = ph - pl
    r1 = pc + 1.1 * rang * 1.0833  # R1 = Close + 1.1 * (High-Low) * 1.0833
    s1 = pc - 1.1 * rang * 1.0833  # S1 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align weekly Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter: current volume / 24-period average volume (24*12h = 12 days)
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
    
    start_idx = max(24, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA50) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below S1 OR trend reversal (price < EMA50)
                if close[i] < s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above R1 OR trend reversal (price > EMA50)
                if close[i] > r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals