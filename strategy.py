#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1D and 1W Pivot R1/S1 breakout with volume confirmation
# - Uses 1D and 1W pivot points (R1, S1) from prior period
# - Long when price breaks above R1 with volume > 1.5x average
# - Short when price breaks below S1 with volume > 1.5x average
# - Exit when price returns to pivot point or opposite level
# - Designed for low frequency (<30 trades/year) to minimize fee drag
# - Works in both bull and bear markets by following price action at key levels

name = "12h_Pivot_R1S1_Breakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
    # Use previous day's values (already completed)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pivot_1d = typical_price_1d.values
    r1_1d = (2 * pivot_1d - df_1d['low'].values)
    s1_1d = (2 * pivot_1d - df_1d['high'].values)
    
    # Align to 12h timeframe (waits for 1D bar to close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1W data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W pivot points
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    pivot_1w = typical_price_1w.values
    r1_1w = (2 * pivot_1w - df_1w['low'].values)
    s1_1w = (2 * pivot_1w - df_1w['high'].values)
    
    # Align to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any pivot data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above R1 from either timeframe with volume
            if ((close[i] > r1_1d_aligned[i] or close[i] > r1_1w_aligned[i]) and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 from either timeframe with volume
            elif ((close[i] < s1_1d_aligned[i] or close[i] < s1_1w_aligned[i]) and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price returns to pivot or goes below S1
            if (close[i] < pivot_1d_aligned[i] or close[i] < pivot_1w_aligned[i] or
                close[i] < s1_1d_aligned[i] or close[i] < s1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price returns to pivot or goes above R1
            if (close[i] > pivot_1d_aligned[i] or close[i] > pivot_1w_aligned[i] or
                close[i] > r1_1d_aligned[i] or close[i] > r1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals