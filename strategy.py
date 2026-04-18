#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume
Hypothesis: Uses weekly pivot levels (R1/S1) from the previous week as support/resistance,
with volume confirmation and price action filters to capture breakouts.
Designed for low trade frequency (<25/year) to minimize fee drag and work in both
bull and bear markets by trading with the weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points and support/resistance levels
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Standard pivot point calculation
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Align weekly levels to daily timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Price action filter: close must be outside the weekly pivot range
    # This avoids whipsaws in ranging markets
    price_action = (close > r1_aligned) | (close < s1_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume confirmation and price above pivot
            if (close[i] > r1_aligned[i] and vol_confirm[i] and 
                close[i] > pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and price below pivot
            elif (close[i] < s1_aligned[i] and vol_confirm[i] and 
                  close[i] < pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below S1 or volume drops significantly
            if (close[i] < s1_aligned[i] or 
                volume[i] < np.mean(volume[max(0, i-5):i+1]) * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above R1 or volume drops significantly
            if (close[i] > r1_aligned[i] or 
                volume[i] < np.mean(volume[max(0, i-5):i+1]) * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0