#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Volume_Trend
Strategy: Daily price breakout above/below weekly Donchian channels (20-week) with volume confirmation and weekly trend filter.
Long: Price breaks above weekly Donchian high(20) + volume > 1.5x 20-day average + price above weekly EMA20
Short: Price breaks below weekly Donchian low(20) + volume > 1.5x 20-day average + price below weekly EMA20
Exit: Price returns to weekly Donchian midpoint
Position size: 0.25
Designed to capture major trend continuations in both bull and bear markets with low trade frequency.
Timeframe: 1d
"""

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
    
    # Calculate weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    high_rolling_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_rolling_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_1w = (high_rolling_1w + low_rolling_1w) / 2
    
    # Align weekly Donchian to daily timeframe
    high_rolling_1w_aligned = align_htf_to_ltf(prices, df_1w, high_rolling_1w)
    low_rolling_1w_aligned = align_htf_to_ltf(prices, df_1w, low_rolling_1w)
    donchian_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation (20-day average on daily)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need at least 20 days for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_rolling_1w_aligned[i]) or 
            np.isnan(low_rolling_1w_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA20
        price_above_ema = close[i] > ema20_1w_aligned[i]
        price_below_ema = close[i] < ema20_1w_aligned[i]
        
        # Breakout conditions (using previous period's levels)
        breakout_up = close[i] > high_rolling_1w_aligned[i-1]
        breakout_down = close[i] < low_rolling_1w_aligned[i-1]
        
        # Return to midpoint (exit condition)
        return_to_mid = abs(close[i] - donchian_mid_1w_aligned[i]) < 0.1 * (high_rolling_1w_aligned[i] - low_rolling_1w_aligned[i])
        
        if position == 0:
            # Long: breakout up + volume filter + price above weekly EMA
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + price below weekly EMA
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to midpoint or break down
            if return_to_mid or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to midpoint or break up
            if return_to_mid or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0