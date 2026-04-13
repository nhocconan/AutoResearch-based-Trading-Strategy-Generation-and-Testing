#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for bias and structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high_20w = rolling_max(high_1w, 20)
    donchian_low_20w = rolling_min(low_1w, 20)
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(
        high_1w - low_1w,
        np.maximum(
            np.abs(high_1w - np.roll(close_1w, 1)),
            np.abs(low_1w - np.roll(close_1w, 1))
        )
    )
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = np.zeros_like(tr_1w)
    for i in range(len(tr_1w)):
        if i < 14:
            atr_1w[i] = np.mean(tr_1w[:i+1]) if i > 0 else tr_1w[i]
        else:
            atr_1w[i] = 0.93 * atr_1w[i-1] + 0.07 * tr_1w[i]
    
    # Get daily data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume spike (volume > 1.5x 20-day average)
    vol_ma_20d = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_ma_20d[i] = np.mean(volume_1d[:i+1]) if i > 0 else volume_1d[i]
        else:
            vol_ma_20d[i] = 0.9 * vol_ma_20d[i-1] + 0.1 * volume_1d[i]
    volume_spike = volume_1d > (1.5 * vol_ma_20d)
    
    # Calculate daily close position within weekly range
    weekly_range = donchian_high_20w - donchian_low_20w
    weekly_range_safe = np.where(weekly_range == 0, 1, weekly_range)
    position_in_weekly = (close_1d - donchian_low_20w) / weekly_range_safe
    
    # Align weekly indicators to 6h timeframe
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    position_in_weekly_aligned = align_htf_to_ltf(prices, df_1w, position_in_weekly)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20w_aligned[i]) or 
            np.isnan(donchian_low_20w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(position_in_weekly_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly bias: price above/below weekly Donchian midpoint
        weekly_mid = (donchian_high_20w_aligned[i] + donchian_low_20w_aligned[i]) / 2
        bullish_bias = close[i] > weekly_mid
        bearish_bias = close[i] < weekly_mid
        
        # Breakout conditions: price breaks weekly Donchian bands with volume
        breakout_up = close[i] > donchian_high_20w_aligned[i] and volume_spike_aligned[i]
        breakout_down = close[i] < donchian_low_20w_aligned[i] and volume_spike_aligned[i]
        
        # Fade conditions: price at extremes of weekly range with volume
        fade_up = (position_in_weekly_aligned[i] > 0.8) and volume_spike_aligned[i]
        fade_down = (position_in_weekly_aligned[i] < 0.2) and volume_spike_aligned[i]
        
        # Entry logic: breakout with bias, fade against bias
        long_entry = breakout_up and bullish_bias
        short_entry = breakout_down and bearish_bias
        long_fade = fade_down and bullish_bias  # fade at top in bullish bias
        short_fade = fade_up and bearish_bias   # fade at bottom in bearish bias
        
        # Exit conditions: opposite signal or loss of bias
        exit_long = position == 1 and (not bullish_bias or breakout_down or fade_up)
        exit_short = position == -1 and (not bearish_bias or breakout_up or fade_down)
        
        # Execute signals
        if (long_entry or long_fade) and position != 1:
            position = 1
            signals[i] = position_size
        elif (short_entry or short_fade) and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "6h_1w_1d_donchian_volume_bias_fade_v1"
timeframe = "6h"
leverage = 1.0