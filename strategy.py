#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1h chop regime filter
    # Long: price breaks above Donchian(20) high + volume > 1.5x 20-period average + chop < 61.8 (trending)
    # Short: price breaks below Donchian(20) low + volume > 1.5x 20-period average + chop < 61.8 (trending)
    # Exit: opposite Donchian breakout or chop > 61.8 (range) to avoid whipsaw
    # Uses Donchian for structure, volume for confirmation, chop for regime
    # Works in bull (buy breakouts) and bear (sell breakdowns) with regime filter
    # Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Get 1h data for chop regime filter (MTF)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 1:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 4h Donchian(20) channels
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike (current volume > 1.5x 20-period average)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_1d)
    
    # Calculate 1h chop regime (Ehlers Chop Index)
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros(len(close))
        for i in range(len(close)):
            if i == 0:
                atr[i] = high[i] - low[i]
            else:
                atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(window)
        return np.where((highest_high - lowest_low) == 0, 50, chop)
    
    chop_values = calculate_chop(high_1h, low_1h, close_1h)
    
    # Align HTF indicators to 4h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1h, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_aligned[i]
        breakout_down = close[i] < lowest_low_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor breakouts)
        trending_regime = chop_aligned[i] < 61.8
        
        # Entry conditions
        long_entry = breakout_up and volume_spike_aligned[i] and trending_regime and position != 1
        short_entry = breakout_down and volume_spike_aligned[i] and trending_regime and position != -1
        
        # Exit conditions: opposite breakout or chop > 61.8 (range)
        exit_long = position == 1 and (breakout_down or chop_aligned[i] > 61.8)
        exit_short = position == -1 and (breakout_up or chop_aligned[i] > 61.8)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_1d_1h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0