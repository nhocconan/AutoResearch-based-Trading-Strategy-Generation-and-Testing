#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > SMA200) and volume confirmation
# Uses Donchian channel breakouts for trend continuation
# Requires price to be above 1d SMA200 for long bias or below for short bias
# Volume confirmation (>1.5x 20-bar average) ensures institutional participation
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: captures strong trends with filter, avoids false signals in weak trends

name = "6h_Donchian20_1dSMA200_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d SMA200 for trend filter
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Calculate Donchian(20) channels on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(sma200_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above 1d SMA200 AND volume confirmation
            if (close[i] > highest_20[i] and close[i] > sma200_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below 1d SMA200 AND volume confirmation
            elif (close[i] < lowest_20[i] and close[i] < sma200_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price closes below Donchian low
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian high
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals