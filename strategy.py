#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d EMA Trend Filter + Volume Spike
Long: Price breaks above Donchian(20) high + 1d EMA up + volume spike
Short: Price breaks below Donchian(20) low + 1d EMA down + volume spike
Exit: Price crosses Donchian(20) midline (10-day avg of high/low)
Donchian captures breakouts, EMA filter ensures higher timeframe trend alignment,
volume spike confirms momentum. Designed for 4h to work in both bull and bear markets.
Target: 80-150 total trades over 4 years (20-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window=20):
    """Calculate Donchian channels: upper, lower, middle"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max()
    lower = pd.Series(low).rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels on 4h (20-period)
    upper, lower, middle = calculate_donchian(high, low, window=20)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d EMA slope for trend filter
    ema_slope = np.diff(ema_34_1d_aligned, prepend=ema_34_1d_aligned[0])
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian calculations
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper + 1d EMA up + volume spike
            if (price > upper[i] and 
                ema_slope[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + 1d EMA down + volume spike
            elif (price < lower[i] and 
                  ema_slope[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian middle
            if price < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian middle
            if price > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0