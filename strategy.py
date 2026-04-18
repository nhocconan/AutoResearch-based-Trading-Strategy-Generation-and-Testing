#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and 1d EMA34 Trend Filter
Uses Donchian(20) breakout from 1d combined with volume confirmation and 1d EMA trend filter.
Designed for low trade frequency with strong edge in both bull and bear markets by
taking breakouts in direction of higher timeframe trend. Focuses on breakout strength
to avoid noise and reduce trade frequency. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for Donchian calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and above 1d EMA
            if (price > upper_channel and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and below 1d EMA
            elif (price < lower_channel and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: reverse signal (break below lower channel)
            if price < lower_channel:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: reverse signal (break above upper channel)
            if price > upper_channel:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_Volume_Spike_1dEMA34"
timeframe = "12h"
leverage = 1.0