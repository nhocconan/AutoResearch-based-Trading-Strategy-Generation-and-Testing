#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
Long when price breaks above 20-bar high AND weekly pivot shows bullish bias (price > weekly R1) AND 6h volume > 1.5x 20-bar avg.
Short when price breaks below 20-bar low AND weekly pivot shows bearish bias (price < weekly S1) AND 6h volume > 1.5x 20-bar avg.
Exit when price touches 10-bar midpoint.
Uses 6h for execution and volume, 1w for pivot bias.
Designed to capture strong breakouts aligned with weekly structure across bull and bear markets.
Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Get 6h data for Donchian channels and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 6h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_6h, donch_mid)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_high = close[i] > donch_high_aligned[i]
        breakout_low = close[i] < donch_low_aligned[i]
        
        # Exit condition: touch 10-bar midpoint (tighter than original)
        touch_mid = abs(close[i] - donch_mid_aligned[i]) < 0.0005 * close[i]  # within 0.05%
        
        # Weekly pivot bias
        weekly_bullish = close[i] > r1_1w_aligned[i]   # price above weekly R1
        weekly_bearish = close[i] < s1_1w_aligned[i]   # price below weekly S1
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and weekly bullish bias
            if (breakout_high and volume_confirmed and weekly_bullish):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and weekly bearish bias
            elif (breakout_low and volume_confirmed and weekly_bearish):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0