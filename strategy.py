#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_R1_S1_Breakout_Volume_ATR_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels, volume, and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for volume filter
    avg_vol = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(avg_vol_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume_1d_aligned[i] > 1.5 * avg_vol_aligned[i]
        
        if position == 0:
            # Long breakout above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short breakdown below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price drops below S1 or ATR-based stop
            if (close[i] < s1_aligned[i] or 
                close[i] < r1_aligned[i] - 1.5 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 or ATR-based stop
            if (close[i] > r1_aligned[i] or 
                close[i] > s1_aligned[i] + 1.5 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals