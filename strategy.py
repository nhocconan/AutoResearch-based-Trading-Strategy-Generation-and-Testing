#!/usr/bin/env python3
"""
6h_Donchian_20_Breakout_1wPivot_Trend_VolumeSpike_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with 1w pivot-based trend filter and volume confirmation capture strong trends while avoiding whipsaws. Weekly pivot direction (price above/below weekly pivot) provides robust trend filter that works in both bull and bear markets. Volume confirmation ensures breakouts have conviction. Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Load 1w data ONCE before loop for HTF trend filter (weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    # Shift by 1 to use previous week's values
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h Donchian(20) breakout levels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 6h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for Donchian and volume MA)
    start_idx = max(donchian_window, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (weekly pivot)
        uptrend = close[i] > weekly_pivot_aligned[i]
        downtrend = close[i] < weekly_pivot_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Long logic: breakout above Donchian high in uptrend with volume
        if uptrend and volume_spike and breakout_up:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below Donchian low in downtrend with volume
        elif downtrend and volume_spike and breakout_down:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_Breakout_1wPivot_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0