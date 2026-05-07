#!/usr/bin/env python3
"""
12h_1D_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: Uses Donchian channel breakout on 1d timeframe for trend direction,
combined with volume confirmation on 12h timeframe to filter false signals.
Trades only in direction of higher timeframe trend to reduce whipsaws.
Designed for low trade frequency (15-25/year) to minimize fee drag while capturing
major trends in both bull and bear markets.
"""

name = "12h_1D_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian Channel (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    upper_20 = np.full_like(high_1d, np.nan)
    lower_20 = np.full_like(low_1d, np.nan)
    
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian bands to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume confirmation: volume > 20-period average (20 * 12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation
            if close[i] > upper_20_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume confirmation
            elif close[i] < lower_20_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below lower Donchian band (contrarian exit)
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above upper Donchian band (contrarian exit)
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals