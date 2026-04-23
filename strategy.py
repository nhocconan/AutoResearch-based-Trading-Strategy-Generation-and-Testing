#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation.
Long when price breaks above 20-bar high + 12h EMA50 uptrend + volume > 1.5x average.
Short when price breaks below 20-bar low + 12h EMA50 downtrend + volume > 1.5x average.
Exit when price crosses 20-bar midpoint or 12h EMA50 trend changes.
Designed for low trade frequency (~15-30/year) to minimize fee drift in both bull and bear markets.
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
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    mid_20 = (high_20 + low_20) / 2
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 12h close for trend direction (use last available 12h bar)
        idx_12h = min(i // 2, len(df_12h) - 1)  # 2x 6h bars = 1 12h bar
        close_12h_current = df_12h['close'].values[idx_12h] if idx_12h < len(df_12h) else np.nan
        ema50_12h_current = ema50_12h_aligned[i]
        
        if np.isnan(close_12h_current) or np.isnan(ema50_12h_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_12h_current > ema50_12h_current
        trend_down = close_12h_current < ema50_12h_current
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: break above 20-bar high + 12h uptrend + volume confirmation
            if (close[i] > high_20[i] and 
                trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-bar low + 12h downtrend + volume confirmation
            elif (close[i] < low_20[i] and 
                  trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 20-bar midpoint or trend changes to down
                if close[i] < mid_20[i] or not trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 20-bar midpoint or trend changes to up
                if close[i] > mid_20[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_12hEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0