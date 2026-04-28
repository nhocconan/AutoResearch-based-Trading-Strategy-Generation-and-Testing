#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotTrend_Volume
Hypothesis: 6-hour Donchian(20) breakouts aligned with weekly pivot trend and volume confirmation. 
Weekly pivot provides institutional context, Donchian provides breakout edge, volume confirms strength.
Works in bull/bear: breakouts capture momentum, weekly pivot filters counter-trend noise.
Targets ~20-40 trades/year by requiring confluence of three filters.
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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week (using prior 5 daily bars)
    # Weekly high = max of prior 5 daily highs
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    # Weekly low = min of prior 5 daily lows  
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    # Weekly close = prior Friday's close (5 days ago)
    weekly_close = pd.Series(df_1d['close']).shift(5).values
    
    # Weekly pivot point and key levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Get 60-period high/low for Donchian channel (approx 20 periods of 6h data)
    # Since we're on 6h timeframe, use 20-period Donchian directly
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/below weekly pivot
        weekly_trend_up = close[i] > weekly_pivot_aligned[i]
        weekly_trend_down = close[i] < weekly_pivot_aligned[i]
        
        # Entry conditions
        # Long: Donchian breakout above weekly R1 + weekly uptrend + volume surge
        long_entry = (high[i] > donchian_high[i] and 
                     close[i] > weekly_r1_aligned[i] and 
                     weekly_trend_up and 
                     volume_surge[i])
        
        # Short: Donchian breakdown below weekly S1 + weekly downtrend + volume surge
        short_entry = (low[i] < donchian_low[i] and 
                      close[i] < weekly_s1_aligned[i] and 
                      weekly_trend_down and 
                      volume_surge[i])
        
        # Exit conditions: reversal signal with volume
        long_exit = (low[i] < donchian_low[i] and 
                    close[i] < weekly_s1_aligned[i] and 
                    volume_surge[i])
        
        short_exit = (high[i] > donchian_high[i] and 
                     close[i] > weekly_r1_aligned[i] and 
                     volume_surge[i])
        
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