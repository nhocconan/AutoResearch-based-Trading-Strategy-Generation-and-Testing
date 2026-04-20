#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_VolumeTrend_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with 1d trend filter and volume confirmation.
Long when price breaks above 20-period 4h high with volume spike and 1d EMA50 > EMA200.
Short when price breaks below 20-period 4h low with volume spike and 1d EMA50 < EMA200.
Uses 1d EMA for trend filter (bull/bear adaptability) and volume spike for institutional confirmation.
Designed for low trade frequency (~25-50/year) to minimize fee drag. Works in bull/bear via trend filter.
"""

name = "4h_1d_Donchian20_Breakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Vectorized rolling max/min
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(20, n):  # min_periods=20
        high_max[i] = np.max(high_4h[i-19:i+1])
        low_min[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h volume average (20-period) for spike detection
    volume_4h = prices['volume'].values
    vol_avg_4h = np.full(n, np.nan)
    for i in range(20, n):  # min_periods=20
        vol_avg_4h[i] = np.mean(volume_4h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if EMA data not available yet
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]):
            continue
            
        current_close = prices['close'].iloc[i]
        current_high = prices['high'].iloc[i]
        current_low = prices['low'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume spike: current volume > 1.8x 20-period average
        vol_spike = (not np.isnan(vol_avg_4h[i]) and 
                     current_volume > 1.8 * vol_avg_4h[i])
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume spike and 1d uptrend
            if (not np.isnan(high_max[i]) and 
                current_high > high_max[i] and vol_spike and
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low with volume spike and 1d downtrend
            elif (not np.isnan(low_min[i]) and 
                  current_low < low_min[i] and vol_spike and
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or trend reverses
            if ((not np.isnan(low_min[i]) and 
                 current_low < low_min[i]) or
                ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or trend reverses
            if ((not np.isnan(high_max[i]) and 
                 current_high > high_max[i]) or
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals