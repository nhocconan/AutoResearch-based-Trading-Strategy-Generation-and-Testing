#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter
    # Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + chop < 61.8 (trending)
    # Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + chop < 61.8 (trending)
    # Exit: price crosses Donchian(20) midpoint
    # Uses 12h primary timeframe for structure, 1d for volume/volatility filters
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume and choppiness filters (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid_12h = (highest_high_12h + lowest_low_12h) / 2.0
    
    # Calculate 1d volume spike filter: current volume > 1.5x 20-period average
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_1d)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first period
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_1d - min_low_1d
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid division by zero
    chopiness = 100 * np.log10(sum_atr_1d / chop_denominator) / np.log10(14)
    chop_filter = chopiness < 61.8  # trending regime
    
    # Align 1d filters to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_12h[i]
        breakout_down = close[i] < lowest_low_12h[i]
        
        # Exit condition: price crosses midpoint
        exit_signal = (position == 1 and close[i] < donchian_mid_12h[i]) or \
                      (position == -1 and close[i] > donchian_mid_12h[i])
        
        # Entry conditions: breakout + volume spike + trending regime
        long_entry = breakout_up and volume_spike_aligned[i] > 0.5 and chop_filter_aligned[i] > 0.5 and position != 1
        short_entry = breakout_down and volume_spike_aligned[i] > 0.5 and chop_filter_aligned[i] > 0.5 and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif exit_signal:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_volume_chop_filter_v1"
timeframe = "12h"
leverage = 1.0