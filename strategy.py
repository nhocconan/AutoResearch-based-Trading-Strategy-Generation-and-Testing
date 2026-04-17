#!/usr/bin/env python3
"""
12h_1d_20_Donchian_Breakout_Volume_Trend_v1
Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA trend filter.
Works in bull (captures breakouts) and bear (avoids false signals via trend filter).
Target: 15-30 trades/year (60-120 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channel
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max20 + low_min20) / 2.0
    
    # Volume confirmation: 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 34)  # Donchian20, volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > high_max20[i-1]
        breakout_down = close[i] < low_min20[i-1]
        
        # Return to midpoint for exit
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.002 * close[i]
        
        if position == 0:
            # Long: breakout up + volume filter + 1d uptrend
            if breakout_up and volume_filter and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + 1d downtrend
            elif breakout_down and volume_filter and close[i] < ema34_1d_aligned[i]:
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

name = "12h_1d_20_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0