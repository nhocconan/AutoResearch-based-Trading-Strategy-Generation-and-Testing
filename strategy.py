#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot Point = (H + L + C)/3
    pivot = (high_w + low_w + close_w) / 3
    # Support 1 = (2*P) - H
    s1 = (2 * pivot) - high_w
    # Resistance 1 = (2*P) - L
    r1 = (2 * pivot) - low_w
    # Support 2 = P - (H - L)
    s2 = pivot - (high_w - low_w)
    # Resistance 2 = P + (H - L)
    r2 = pivot + (high_w - low_w)
    
    # Get daily data for volume and price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    vol_d = df_1d['volume'].values
    close_d = df_1d['close'].values
    
    # Volume spike detection (current day > 1.5x 20-day average)
    vol_series = pd.Series(vol_d)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_d > (vol_ma * 1.5)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Align daily volume spike to 6h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to weekly pivot levels
        at_pivot = abs(close[i] - pivot_aligned[i]) < (pivot_aligned[i] * 0.005)  # Within 0.5% of pivot
        near_r1 = abs(close[i] - r1_aligned[i]) < (r1_aligned[i] * 0.005)  # Within 0.5% of R1
        near_s1 = abs(close[i] - s1_aligned[i]) < (s1_aligned[i] * 0.005)  # Within 0.5% of S1
        near_r2 = abs(close[i] - r2_aligned[i]) < (r2_aligned[i] * 0.005)  # Within 0.5% of R2
        near_s2 = abs(close[i] - s2_aligned[i]) < (s2_aligned[i] * 0.005)  # Within 0.5% of S2
        
        # Volume confirmation
        vol_confirmed = vol_spike_aligned[i]
        
        # Entry logic: fade at S1/R1, breakout at S2/R2 with volume
        long_entry = (near_s1 or near_s2) and vol_confirmed and close[i] > close[i-1]  # Bounce off support with volume
        short_entry = (near_r1 or near_r2) and vol_confirmed and close[i] < close[i-1]  # Reject resistance with volume
        
        # Exit on opposite signal or when price moves away from pivot area
        exit_long = position == 1 and (close[i] < pivot_aligned[i] * 0.995 or close[i] > r2_aligned[i] * 1.005)
        exit_short = position == -1 and (close[i] > pivot_aligned[i] * 1.005 or close[i] < s2_aligned[i] * 0.995)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_weekly_pivot_volume_reversion"
timeframe = "6h"
leverage = 1.0