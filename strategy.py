#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_R1_S1_Breakout_VolumeATR_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for ATR and pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily pivot point (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate R1 and S1 using standard pivot formula
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily values to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14_aligned[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        atr_threshold = 0.5 * atr  # Minimum price move for breakout
        
        if position == 0:
            # Long: break above R1 with volume and sufficient momentum
            if price > r1 + atr_threshold and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and sufficient momentum
            elif price < s1 - atr_threshold and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below pivot or ATR-based trailing stop
            if price < pivot or price < (high[i] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot or ATR-based trailing stop
            if price > pivot or price > (low[i] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals