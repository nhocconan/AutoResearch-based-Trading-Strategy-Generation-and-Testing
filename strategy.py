#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_Volume_Spike
Strategy: Daily breakout above weekly Donchian high/low with volume spike and 1w trend filter.
Long: Price breaks above weekly Donchian high(20) + volume > 2x average + price above 1w EMA34
Short: Price breaks below weekly Donchian low(20) + volume > 2x average + price below 1w EMA34
Exit: Price returns to weekly Donchian midpoint
Position size: 0.25
Designed to capture major breakouts aligned with weekly trend in both bull and bear markets.
Timeframe: 1d
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
    
    # Calculate weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    high_rolling = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_rolling + low_rolling) / 2
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation (20-period MA on daily)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_rolling[i]) or 
            np.isnan(low_rolling[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_rolling[i-1]  # break above previous period high
        breakout_down = close[i] < low_rolling[i-1]  # break below previous period low
        
        # Reversion to midpoint
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.1 * (high_rolling[i] - low_rolling[i])
        
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

name = "1d_WeeklyDonchian20_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0