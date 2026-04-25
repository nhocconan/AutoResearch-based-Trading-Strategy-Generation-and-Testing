#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivotDir
Hypothesis: Trade 6h Donchian(20) breakouts only when 1d EMA50 confirms trend (price above/below EMA) and price is on favorable side of weekly Camarilla H3/L3 levels (from prior week). Weekly pivot provides institutional bias; Donchian breakout captures momentum; EMA50 filter avoids counter-trend whipsaws. Uses discrete sizing 0.25. Target 12-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h (completed 1d bar only)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h (completed 1d bar only)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for Camarilla H3/L3 levels (prior week)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla H3/L3 levels
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_weekly_range = (high_1w - low_1w) * 1.1 / 4.0
    camarilla_H3 = close_1w + camarilla_weekly_range
    camarilla_L3 = close_1w - camarilla_weekly_range
    
    # Align weekly Camarilla levels to 6h (completed 1w bar only)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA50 (50), weekly data
    start_idx = max(50, 20)  # 50 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + above 1d EMA50 + above weekly H3 (bullish bias)
            long_setup = (close[i] > highest_20_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         (close[i] > camarilla_H3_aligned[i])
            # Short: price breaks below Donchian lower + below 1d EMA50 + below weekly L3 (bearish bias)
            short_setup = (close[i] < lowest_20_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          (close[i] < camarilla_L3_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Donchian lower OR below 1d EMA50
            if (close[i] < lowest_20_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Donchian upper OR above 1d EMA50
            if (close[i] > highest_20_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivotDir"
timeframe = "6h"
leverage = 1.0