#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter
Strategy: 6h breakout at weekly pivot R1/S1 with volume confirmation.
- Long when price breaks above weekly R1 + volume > 1.5x 20-period avg
- Short when price breaks below weekly S1 + volume > 1.5x 20-period avg
- Exit when price returns to weekly pivot point or opposite breakout occurs
- Position size: ±0.25
- Uses weekly pivot levels (calculated from prior week) for structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot, R1, S1 for each week
    weekly_pivot_val, weekly_r1, weekly_s1 = weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly levels to 6h timeframe (using previous week's values)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot_val)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Volume confirmation (20-period MA on 6h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 1)  # volume MA20 and weekly data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions at weekly R1/S1
        breakout_up = close[i] > r1_aligned[i-1]  # break above weekly R1
        breakout_down = close[i] < s1_aligned[i-1]  # break below weekly S1
        
        # Return to weekly pivot for exit
        return_to_pivot = abs(close[i] - pivot_aligned[i]) < 0.005 * close[i]  # within 0.5% of pivot
        
        if position == 0:
            # Long: breakout above R1 + volume filter
            if breakout_up and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter
            elif breakout_down and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot or opposite breakout
            if return_to_pivot or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot or opposite breakout
            if return_to_pivot or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0