#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_v2"
timeframe = "1d"
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points using prior week's data
    # We need previous week's data, not current week
    high_prev = np.roll(high_1w, 1)
    low_prev = np.roll(low_1w, 1)
    close_prev = np.roll(close_1w, 1)
    
    # First element will be invalid due to roll, we'll handle with isnan check
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    
    # Align weekly pivot levels to daily timeframe
    pivot_d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_d = align_htf_to_ltf(prices, df_1w, r1)
    s1_d = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Need 20 for volume MA + 1 for weekly data
    
    for i in range(start_idx, n):
        if np.isnan(pivot_d[i]) or np.isnan(r1_d[i]) or np.isnan(s1_d[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above R1 with volume
            if price > r1_d[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume
            elif price < s1_d[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot
            if price < pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot
            if price > pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals