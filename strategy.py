#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 1d EMA Trend Filter
Uses Donchian(20) channel breakouts from 4h price action combined with volume confirmation
and 1d EMA trend filter. Designed for low trade frequency with strong edge in both bull
and bear markets by taking breakouts in direction of higher timeframe trend.
Focuses on clear breakouts to avoid noise and reduce trade frequency.
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
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike and above 1d EMA
            if (price > upper_channel and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower channel with volume spike and below 1d EMA
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

name = "4h_Donchian_Breakout_Volume_1dEMA34"
timeframe = "4h"
leverage = 1.0