#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA200_Trend_Volume
Hypothesis: Breakouts from daily Camarilla R1/S1 levels with 12h EMA200 trend filter and volume spike confirmation.
The 12h EMA200 provides a robust trend filter that adapts to bull/bear markets while being less prone to whipsaw than shorter EMAs.
Volume spike (>2x 48-period average) confirms breakout strength. Designed for low trade frequency (19-50/year) to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hEMA200_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA200
    ema_200_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 200:
        ema_200_12h[199] = np.mean(close_12h[0:200])
        for i in range(200, len(close_12h)):
            ema_200_12h[i] = (ema_200_12h[i-1] * 199 + close_12h[i]) / 200
    
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
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
    
    # Align daily Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike filter: current volume / 48-period average volume (48*4h = 8 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 48:
        vol_ma[47] = np.mean(volume[0:48])
        for i in range(48, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 47 + volume[i]) / 48
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(48, 200)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA200) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_200_12h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND downtrend (price < EMA200) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_200_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below S1 OR trend reversal (price < EMA200)
                if close[i] < s1_aligned[i] or close[i] < ema_200_12h_aligned[i]:
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
                # Exit short: price breaks above R1 OR trend reversal (price > EMA200)
                if close[i] > r1_aligned[i] or close[i] > ema_200_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals