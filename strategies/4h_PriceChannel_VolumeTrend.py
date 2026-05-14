#!/usr/bin/env python3
"""
4h_PriceChannel_VolumeTrend
Strategy: 4-hour price channel breakout with volume confirmation and 1d trend filter.
Long: Price breaks above 4h Donchian high(20) + volume > 1.5x average + price above 1d EMA34
Short: Price breaks below 4h Donchian low(20) + volume > 1.5x average + price below 1d EMA34
Exit: Price returns to 4h Donchian midpoint
Position size: 0.30
Designed to capture breakouts aligned with daily trend in both bull and bear markets.
Timeframe: 4h
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
    
    # Calculate Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_rolling + low_rolling) / 2
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_rolling[i]) or 
            np.isnan(low_rolling[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_rolling[i-1]  # break above previous period high
        breakout_down = close[i] < low_rolling[i-1]  # break below previous period low
        
        # Reversion to midpoint
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.1 * (high_rolling[i] - low_rolling[i])
        
        if position == 0:
            # Long: breakout up + volume filter + price above EMA
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.30
                position = 1
            # Short: breakout down + volume filter + price below EMA
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: return to midpoint or break down
            if return_to_mid or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: return to midpoint or break up
            if return_to_mid or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_PriceChannel_VolumeTrend"
timeframe = "4h"
leverage = 1.0