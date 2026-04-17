#!/usr/bin/env python3
"""
6h Monthly Pivot Breakout with Volume Confirmation
Long: Break above monthly R1 + volume > 2x 6m avg volume
Short: Break below monthly S1 + volume > 2x 6m avg volume
Exit: Price crosses monthly pivot point or opposite signal
Uses monthly pivots for long-term structure, volume for confirmation.
Works in bull/bear by following monthly pivot structure.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get monthly data for pivot calculation
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 3:
        return np.zeros(n)
    
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    monthly_close = df_monthly['close'].values
    
    # Calculate monthly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    monthly_pivot = (monthly_high + monthly_low + monthly_close) / 3.0
    monthly_r1 = 2 * monthly_pivot - monthly_low
    monthly_s1 = 2 * monthly_pivot - monthly_high
    
    # Align monthly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_monthly, monthly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_monthly, monthly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_monthly, monthly_s1)
    
    # Calculate 6m volume SMA(20) for volume filter
    vol_sma_6m = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 20)  # need volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_sma_6m[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_6m[i]
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        if position == 0:
            # Long: Break above monthly R1 with volume confirmation
            if price > r1 and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below monthly S1 with volume confirmation
            elif price < s1 and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below monthly pivot or opposite signal
            if price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above monthly pivot or opposite signal
            if price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MonthlyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0