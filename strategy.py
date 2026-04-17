#!/usr/bin/env python3
"""
1d_Donchian_20_Breakout_Volume_Trend_HTF
Strategy: 1d Donchian(20) breakout with volume confirmation and 1w trend filter.
- Long when price breaks above 20-day high + volume > 1.8x 20-day avg + 1w close > 1w EMA34
- Short when price breaks below 20-day low + volume > 1.8x 20-day avg + 1w close < 1w EMA34
- Exit when price returns to 20-day midpoint or opposite breakout occurs
- Position size: ±0.25
- Uses 1d timeframe as primary with 1w for trend filter
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
    
    # Calculate 20-day Donchian channels and midpoint
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max20 + low_min20) / 2.0
    
    # Volume confirmation (20-day MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 34)  # Donchian20, volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-day average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > high_max20[i-1]  # break above 20-day high
        breakout_down = close[i] < low_min20[i-1]  # break below 20-day low
        
        # Return to midpoint for exit
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.002 * close[i]  # within 0.2% of midpoint
        
        if position == 0:
            # Long: breakout up + volume filter + 1w uptrend
            if breakout_up and volume_filter and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + 1w downtrend
            elif breakout_down and volume_filter and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to midpoint or opposite breakout
            if return_to_mid or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to midpoint or opposite breakout
            if return_to_mid or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_20_Breakout_Volume_Trend_HTF"
timeframe = "1d"
leverage = 1.0