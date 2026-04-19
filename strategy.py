#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week pivot-based breakout strategy
# Uses weekly pivot levels (R2/S2) for structural breakouts with volume confirmation
# Weekly timeframe provides robust support/resistance that works in both bull/bear markets
# Volume filter ensures breakouts have conviction, reducing false signals
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_Weekly_Pivot_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R2 = P + (H-L), S2 = P - (H-L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above R2 with volume
            if price > r2 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume
            elif price < s2 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below weekly pivot (structure break)
            if price < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above weekly pivot (structure break)
            if price > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals