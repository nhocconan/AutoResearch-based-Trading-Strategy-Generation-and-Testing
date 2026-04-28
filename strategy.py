#SBK
#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotTrend_Volume
Hypothesis: 6-hour Donchian channel breakouts (20-period) aligned with weekly pivot trend direction and volume confirmation. 
Weekly pivot trend uses price position relative to weekly pivot point (PP) and R1/S1 levels. 
Volume surge confirms breakout strength. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by trading with weekly pivot trend direction.
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
    
    # Get weekly data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = (2 * pp) - prev_week_low
    s1 = (2 * pp) - prev_week_high
    
    # Trend logic: price > R1 = bullish, price < S1 = bearish, between = neutral
    trend_up = close > r1
    trend_down = close < s1
    
    # Get daily data for Donchian calculation (using higher resolution for breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from daily data
    # Upper band = 20-period high
    # Lower band = 20-period low
    donchian_upper = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align all higher timeframe data to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above Donchian upper + weekly bullish trend (price > R1) + volume surge
        long_entry = (close[i] > donchian_upper_aligned[i] and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: price breaks below Donchian lower + weekly bearish trend (price < S1) + volume surge
        short_entry = (close[i] < donchian_lower_aligned[i] and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit on opposite Donchian level break with volume surge
        long_exit = close[i] < donchian_lower_aligned[i] and volume_surge[i]
        short_exit = close[i] > donchian_upper_aligned[i] and volume_surge[i]
        
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

name = "6h_Donchian20_WeeklyPivotTrend_Volume"
timeframe = "6h"
leverage = 1.0