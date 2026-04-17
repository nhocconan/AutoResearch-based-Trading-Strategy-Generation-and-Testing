#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian channels (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe
    upper_20_1d = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_1d = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume filter: current volume > 1.3 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d[i]) or np.isnan(upper_20_1d[i]) or np.isnan(lower_20_1d[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume and weekly uptrend
            if (close[i] > upper_20_1d[i] and volume_filter and close[i] > ema34_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume and weekly downtrend
            elif (close[i] < lower_20_1d[i] and volume_filter and close[i] < ema34_1d[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Donchian lower or weekly trend turns down
            if close[i] < lower_20_1d[i] or close[i] < ema34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Donchian upper or weekly trend turns up
            if close[i] > upper_20_1d[i] or close[i] > ema34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA34_Donchian20_VolumeFilter"
timeframe = "1d"
leverage = 1.0