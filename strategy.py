#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for HL2 pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate HL2 (typical price) for pivot calculation
    hl2_1d = (high_1d + low_1d) / 2.0
    
    # Previous day's HL2 for pivot point
    hl2_prev = np.roll(hl2_1d, 1)
    hl2_prev[0] = hl2_1d[0]  # First value
    
    # Calculate pivot point as previous day's HL2
    pivot_1d = hl2_prev
    
    # Calculate R1 and S1 (using HL2)
    r1_1d = 2.0 * pivot_1d - low_1d
    s1_1d = 2.0 * pivot_1d - high_1d
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and weekly uptrend
            if (price > r1_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                price > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and weekly downtrend
            elif (price < s1_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  price < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot or volume dries up
            if price < pivot_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot or volume dries up
            if price > pivot_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HL2Pivot_R1S1_Breakout_VolumeWeeklyTrend"
timeframe = "6h"
leverage = 1.0