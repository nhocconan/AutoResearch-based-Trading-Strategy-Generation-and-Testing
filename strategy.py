#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_Power_Zone"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points (calculated from prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, etc.
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - high_w
    s1 = 2 * pivot - low_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe (delayed by 1 week for completion)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Power Zone: Between S1 and R1 (value area)
    # Breakout above R1 with volume = long, breakdown below S1 with volume = short
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike
            long_breakout = (close[i] > r1_aligned[i] and 
                           close[i-1] <= r1_aligned[i-1] and  # just broke above
                           volume_spike[i])
            
            # Short: price breaks below S1 with volume spike
            short_breakdown = (close[i] < s1_aligned[i] and 
                             close[i-1] >= s1_aligned[i-1] and  # just broke below
                             volume_spike[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakdown:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price returns to pivot zone (below R1) or breaks S2 (stop)
            if close[i] < r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price returns to pivot zone (above S1) or breaks R2 (stop)
            if close[i] > s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals