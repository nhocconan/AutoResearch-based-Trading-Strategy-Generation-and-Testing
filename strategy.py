#!/usr/bin/env python3
"""
12h_1D_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Daily Donchian(20) breakouts with volume confirmation capture sustained moves in both bull and bear markets. The 12h timeframe reduces noise and the 20-period lookback provides robust support/resistance levels. Volume confirmation ensures breakouts have institutional participation. Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Calculate daily Donchian channels (20-period)
    upper = np.full_like(high_daily, np.nan)
    lower = np.full_like(low_daily, np.nan)
    
    for i in range(len(high_daily)):
        if i >= 19:
            upper[i] = np.max(high_daily[i-19:i+1])
            lower[i] = np.min(low_daily[i-19:i+1])
    
    # Align daily Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long breakout: price breaks above upper band with volume
            if price > upper_band and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower band with volume
            elif price < lower_band and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to lower band (mean reversion) or breaks below lower band
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to upper band (mean reversion) or breaks above upper band
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Donchian20_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0