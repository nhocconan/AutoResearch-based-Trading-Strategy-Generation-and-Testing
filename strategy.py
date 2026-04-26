#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) 
provide institutional structure awareness. Combined with 1d EMA50 trend filter and volume confirmation, 
this strategy captures high-probability breakouts in both bull and bear markets. 
Weekly pivots act as key support/resistance levels where price often reacts. 
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data ONCE before loop for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values  # needed for more accurate pivot
    
    # Standard pivot point calculation: P = (H + L + C) / 3
    # Using previous week's data to avoid look-ahead
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h Donchian(20) channels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 6h volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian/volume MA)
    start_idx = max(50, donchian_window, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
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
        
        # 1d trend filter (EMA50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Weekly pivot direction filter
        above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Long logic: Donchian breakout above with uptrend, volume, and price above weekly pivot
        if uptrend and volume_spike and breakout_up and above_weekly_pivot:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Donchian breakdown below with downtrend, volume, and price below weekly pivot
        elif downtrend and volume_spike and breakout_down and below_weekly_pivot:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR Donchian opposite breakout
        elif position == 1 and (not uptrend or breakout_down):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or breakout_up):
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

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0