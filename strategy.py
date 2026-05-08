#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Stochastic_Trend_Filtered_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend and stochastic
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Stochastic Oscillator (14,3,3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1, denominator)
    
    k_percent = 100 * ((close_1d - lowest_low) / denominator)
    # Smooth K with 3-period SMA
    k_smooth = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    # D is 3-period SMA of smoothed K
    d_percent = pd.Series(k_smooth).rolling(window=3, min_periods=3).mean().values
    
    # Align to 4h timeframe
    k_aligned = align_htf_to_ltf(prices, df_1d, k_smooth)
    d_aligned = align_htf_to_ltf(prices, df_1d, d_percent)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Shift to get previous bar's channels (no look-ahead)
    upper_channel_prev = np.roll(upper_channel, 1)
    lower_channel_prev = np.roll(lower_channel, 1)
    upper_channel_prev[0] = np.nan
    lower_channel_prev[0] = np.nan
    
    # Volume spike detection: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(k_aligned[i]) or np.isnan(d_aligned[i]) or 
            np.isnan(upper_channel_prev[i]) or np.isnan(lower_channel_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        k_val = k_aligned[i]
        d_val = d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel, stoch bullish crossover, volume spike
            if (close[i] > upper_channel_prev[i] and 
                k_val > d_val and k_val < 80 and  # Not overbought
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, stoch bearish crossover, volume spike
            elif (close[i] < lower_channel_prev[i] and 
                  k_val < d_val and k_val > 20 and  # Not oversold
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR stoch bearish crossover
            if (close[i] < lower_channel_prev[i] or 
                (k_val < d_val and k_val < 20)):  # Oversold exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel OR stoch bullish crossover
            if (close[i] > upper_channel_prev[i] or 
                (k_val > d_val and k_val > 80)):  # Overbought exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals