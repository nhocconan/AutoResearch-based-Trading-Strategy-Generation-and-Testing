#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate daily pivot points from 12h data (using previous day's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_12h + low_12h + close_12h) / 3.0
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - low_12h
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - high_12h
    # Resistance 2 = P + (H - L)
    r2 = pivot + (high_12h - low_12h)
    # Support 2 = P - (H - L)
    s2 = pivot - (high_12h - low_12h)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Load 6h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_filter = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above R2 with volume (breakout continuation)
            if price > r2_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume (breakout continuation)
            elif price < s2_val and vol_filter:
                signals[i] = -0.25
                position = -1
            # Long fade: price touches S1 and reverses up with volume
            elif price <= s1_val and vol_filter and i > 100 and close[i-1] > s1_val:
                signals[i] = 0.25
                position = 1
            # Short fade: price touches R1 and reverses down with volume
            elif price >= r1_val and vol_filter and i > 100 and close[i-1] < r1_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on break below S1 (mean reversion failure) or volume drop
                if price < s1_val or not vol_filter:
                    exit_signal = True
                # Exit on reaching R2 (take profit)
                elif price >= r2_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on break above R1 (mean reversion failure) or volume drop
                if price > r1_val or not vol_filter:
                    exit_signal = True
                # Exit on reaching S2 (take profit)
                elif price <= s2_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_CamarillaPivot_S1R1_S2R2_BreakoutFade"
timeframe = "6h"
leverage = 1.0