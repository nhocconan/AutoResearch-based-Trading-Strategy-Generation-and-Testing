#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 12h EMA Trend Filter
Uses Donchian(20) breakout from 4h combined with volume confirmation and 12h EMA trend filter.
Designed to capture breakouts in direction of higher timeframe trend with volume confirmation.
Focuses on low trade frequency and strong edge in both bull and bear markets.
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
    
    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        ema_trend = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and above 12h EMA
            if (price > upper_channel and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian with volume spike and below 12h EMA
            elif (price < lower_channel and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
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

name = "4h_Donchian_Breakout_Volume_12hEMA34"
timeframe = "4h"
leverage = 1.0