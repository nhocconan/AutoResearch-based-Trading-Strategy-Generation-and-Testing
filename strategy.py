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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1w data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Donchian channels (20 periods)
    highest_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_20_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20_1w)
    
    # Calculate weekly volume average (20 periods)
    avg_volume_20_1w = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values  # using 6h volume for weekly avg
    avg_volume_20_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_20_1w)
    
    # Calculate daily ATR (14 periods) for volatility filter
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-day average volume for spike detection
    avg_volume_6_6h = pd.Series(volume_6h).rolling(window=6, min_periods=6).mean().values
    volume_ratio = volume_6h / avg_volume_6_6h  # current volume / 6-period average
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(avg_volume_20_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        vol_ratio = volume_ratio[i]
        atr_1d_val = atr_1d_aligned[i]
        donchian_high = donchian_high_1w_aligned[i]
        donchian_low = donchian_low_1w_aligned[i]
        
        # Dynamic thresholds based on volatility
        upper_threshold = donchian_high + 0.5 * atr_1d_val
        lower_threshold = donchian_low - 0.5 * atr_1d_val
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly Donchian low
            if price < donchian_low:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly Donchian high
            if price > donchian_high:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly Donchian high with volume surge
            if (price > upper_threshold) and (vol_ratio > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below weekly Donchian low with volume surge
            elif (price < lower_threshold) and (vol_ratio > 2.0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchianBreakout_VolumeSurge"
timeframe = "6h"
leverage = 1.0