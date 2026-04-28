#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_1dTrend_Filter
Hypothesis: On 6-hour timeframe, enter long when price breaks above 1d Donchian(20) high with weekly pivot support and 1d uptrend, short when price breaks below 1d Donchian(20) low with weekly pivot resistance and 1d downtrend. Uses weekly pivot as dynamic support/resistance and 1d trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) to minimize fee decay in both bull and bear markets. Weekly pivots adapt to volatility, working well in trending and ranging conditions.
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
    
    # Get 1d data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d trend: bullish when close > Donchian mid, bearish when close < Donchian mid
    donchian_mid = (donchian_high + donchian_low) / 2
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    one_d_uptrend = close > donchian_mid_aligned
    one_d_downtrend = close < donchian_mid_aligned
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - high_1w
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - low_1w
    # Support 2 = P - (H - L)
    s2 = pivot - (high_1w - low_1w)
    # Resistance 2 = P + (H - L)
    r2 = pivot + (high_1w - low_1w)
    
    # Align weekly pivot points to 6m timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 1d trend alignment and volume surge
        long_entry = (close[i] > donchian_high_aligned[i] and 
                     close[i] > s1_aligned[i] and  # Price above weekly S1 (support)
                     one_d_uptrend[i] and 
                     volume_surge[i])
        short_entry = (close[i] < donchian_low_aligned[i] and 
                      close[i] < r1_aligned[i] and  # Price below weekly R1 (resistance)
                      one_d_downtrend[i] and 
                      volume_surge[i])
        
        # Exit when price returns to weekly pivot (mean reversion to pivot)
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0