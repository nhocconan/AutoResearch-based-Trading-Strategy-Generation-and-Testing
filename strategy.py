#!/usr/bin/env python3
name = "12h_1w_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:  # Need 20 periods for Donchian + 1 for shift
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter using 20-period EMA
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to 12h timeframe
    upper_band = align_htf_to_ltf(prices, df_1w, high_20)
    lower_band = align_htf_to_ltf(prices, df_1w, low_20)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily volume filter (volume spike)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume 20-period average
    vol_ma20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Align current volume to compare with daily average
    vol_current_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema20_aligned[i]) or np.isnan(vol_ma20_aligned[i]) or
            np.isnan(vol_current_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper band in uptrend with volume surge
            if (close[i] > upper_band[i] and 
                close[i] > ema20_aligned[i] and 
                vol_current_aligned[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower band in downtrend with volume surge
            elif (close[i] < lower_band[i] and 
                  close[i] < ema20_aligned[i] and 
                  vol_current_aligned[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly Donchian lower band or trend turns down
            if (close[i] < lower_band[i] or close[i] < ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above weekly Donchian upper band or trend turns up
            if (close[i] > upper_band[i] or close[i] > ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals