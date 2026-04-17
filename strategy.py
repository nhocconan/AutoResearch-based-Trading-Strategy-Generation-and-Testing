# 6h_WeeklyPivot_R2_S2_Breakout_Volume
# Breakout at weekly R2/S2 with volume confirmation on 6h timeframe
# Weekly pivot provides strong institutional levels; volume confirms breakout strength
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Target: 20-60 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Shift to use previous week's pivots (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_prev)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_prev)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_prev)
    
    # Volume confirmation: current volume > 2.0 * 12-period average (6h * 12 = 3d)
    volume_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need weekly data and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma12[i]) or 
            np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 12-period average
        volume_filter = volume[i] > (2.0 * volume_ma12[i])
        
        if position == 0:
            # Long: price breaks above R2 with volume (strong breakout)
            if close[i] > r2_6h[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume (strong breakdown)
            elif close[i] < s2_6h[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1
            if close[i] < r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1
            if close[i] > s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2_S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0