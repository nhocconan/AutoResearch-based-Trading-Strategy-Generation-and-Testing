#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses Donchian breakout (20-period high/low) for trend entry, filtered by weekly pivot direction
# and volume spike. Only takes long when price breaks above Donchian high AND weekly pivot is bullish
# and volume spike. Only takes short when price breaks below Donchian low AND weekly pivot is bearish
# and volume spike. Exits when price returns to Donchian midpoint or weekly trend reverses.
# Weekly pivot calculated from weekly high/low/close: PP = (H+L+C)/3, R2 = PP + (H-L), S2 = PP - (H-L)
# Target: 15-30 trades per year with position size 0.25 for balanced risk/return in both bull and bear markets.

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot levels: PP = (H+L+C)/3, R2 = PP + (H-L), S2 = PP - (H-L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_range = high_1w - low_1w
    weekly_r2 = weekly_pivot + weekly_range  # Resistance 2
    weekly_s2 = weekly_pivot - weekly_range  # Support 2
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Donchian channel (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Volume spike: current volume > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + weekly pivot bullish (above R2) + volume spike
            if (close[i] > high_20[i] and 
                weekly_pivot_aligned[i] > weekly_r2_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + weekly pivot bearish (below S2) + volume spike
            elif (close[i] < low_20[i] and 
                  weekly_pivot_aligned[i] < weekly_s2_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR weekly pivot turns bearish
            if (close[i] < donchian_mid[i]) or (weekly_pivot_aligned[i] < weekly_r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR weekly pivot turns bullish
            if (close[i] > donchian_mid[i]) or (weekly_pivot_aligned[i] > weekly_s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals