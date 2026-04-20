#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === Weekly High/Low/Close for Pivot Calculation ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's data to calculate pivot
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Weekly pivot point (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Weekly Camarilla levels (R1/S1 for mean reversion, R2/S2 for breakout)
    r1 = pivot + (range_val * 1.1 / 12)  # Strong resistance
    s1 = pivot - (range_val * 1.1 / 12)  # Strong support
    r2 = pivot + (range_val * 1.1 / 6)   # Breakout level
    s2 = pivot - (range_val * 1.1 / 6)   # Breakdown level
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(r2_val) or 
            np.isnan(s2_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion at R1/S1
            if close_val < r1_val and close_val > r1_val * 0.995 and vol_ratio_val > 1.3:
                # Near R1, short with volume
                signals[i] = -0.25
                position = -1
            elif close_val > s1_val and close_val < s1_val * 1.005 and vol_ratio_val > 1.3:
                # Near S1, long with volume
                signals[i] = 0.25
                position = 1
            # Breakout at R2/S2
            elif close_val > r2_val and vol_ratio_val > 1.8:
                # Strong break above R2
                signals[i] = 0.25
                position = 1
            elif close_val < s2_val and vol_ratio_val > 1.8:
                # Strong break below S2
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot or stop at S1
            if close_val < pivot_val or close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot or stop at R1
            if close_val > pivot_val or close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals