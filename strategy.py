#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter_v1"
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
    
    # Get 1d data for Pivot Points and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Pivot Points (R1, S1, R2, S2) from previous day
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    pivot_1d = (H_1d + L_1d + C_1d) / 3.0
    range_1d = H_1d - L_1d
    r1_1d = 2 * pivot_1d - L_1d
    s1_1d = 2 * pivot_1d - H_1d
    r2_1d = pivot_1d + range_1d
    s2_1d = pivot_1d - range_1d
    
    # Align Pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate ATR(14) on 1d for volatility filter
    tr1 = H_1d - L_1d
    tr2 = np.abs(H_1d - np.roll(C_1d, 1))
    tr3 = np.abs(L_1d - np.roll(C_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 (no previous close)
    tr[0] = 0
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: 6h volume > 1.5x 20-period average of 6h volume
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 20-period average
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > 1.5 * vol_ma_6h[i]
        
        # ATR filter: current ATR > 0.5x 20-period average ATR (avoid low volatility)
        atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        atr_filter = atr_1d_aligned[i] > 0.5 * atr_ma_20[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and volatility
            if (close[i] > r1_1d_aligned[i] and 
                volume_filter and 
                atr_filter):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and volatility
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_filter and 
                  atr_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot or breaks S1 (stop)
            if (close[i] <= pivot_1d_aligned[i] or 
                close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot or breaks R1 (stop)
            if (close[i] >= pivot_1d_aligned[i] or 
                close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals