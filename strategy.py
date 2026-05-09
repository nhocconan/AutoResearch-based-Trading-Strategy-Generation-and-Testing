#!/usr/bin/env python3
name = "1D_WeeklyDonchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 20-week Donchian channel (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_high = np.full_like(high_1w, np.nan)
    lowest_low = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        highest_high[i] = np.max(high_1w[i-20:i])
        lowest_low[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to daily
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # Get daily data for 50-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    volume_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > volume_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if weekly Donchian or EMA data not ready
        if np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + above weekly EMA50 + volume confirmation
            if close[i] > highest_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + below weekly EMA50 + volume confirmation
            elif close[i] < lowest_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low OR below weekly EMA50
            if close[i] < lowest_low_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high OR above weekly EMA50
            if close[i] > highest_high_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals