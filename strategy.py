#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with volume confirmation
    # Long: price breaks above H3 (resistance) AND volume > 1.5x 20-period average
    # Short: price breaks below L3 (support) AND volume > 1.5x 20-period average
    # Exit: price returns to PIVOT point or volume drops below average
    # Using 12h for signal direction (Camarilla pivots), 12h only for entry timing
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivots (based on previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    # RANGE = H - L
    range_12h = high_12h - low_12h
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_12h = close_12h + range_12h * 1.1 / 4
    l3_12h = close_12h - range_12h * 1.1 / 4
    
    # Align 12h Camarilla levels to 12h (wait for completed 12h bar)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + volume
        long_entry = (close[i] > h3_12h_aligned[i]) and vol_confirm
        short_entry = (close[i] < l3_12h_aligned[i]) and vol_confirm
        
        # Exit logic: return to pivot or volume dry-up
        long_exit = (close[i] < pivot_12h_aligned[i]) or not vol_confirm
        short_exit = (close[i] > pivot_12h_aligned[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0