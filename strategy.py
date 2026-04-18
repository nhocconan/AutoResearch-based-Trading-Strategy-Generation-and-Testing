#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot trend filter and volume confirmation.
# Long: price breaks above Donchian(20) high AND price above weekly pivot S1 AND volume > 2x 20-period avg.
# Short: price breaks below Donchian(20) low AND price below weekly pivot R1 AND volume > 2x 20-period avg.
# Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
# Volume surge confirms breakout conviction. Works in both bull/bear by filtering with weekly pivot.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_Donchian20_WeeklyPivot_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (ONCE before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    # Using previous week's data to avoid look-ahead
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    
    # Align weekly pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < low_20[i-1]  # Break below previous Donchian low
        
        if position == 0:
            # Long: upward breakout AND price above weekly S1 AND volume spike
            if breakout_up and close[i] > s1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout AND price below weekly R1 AND volume spike
            elif breakout_down and close[i] < r1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend reverses (below S1)
            if close[i] < low_20[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend reverses (above R1)
            if close[i] > high_20[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals