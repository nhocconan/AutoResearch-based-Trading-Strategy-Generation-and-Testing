#!/usr/bin/env python3
# 6h_1w_Donchian_Breakout_TrendFollow_VolumeFilter
# Hypothesis: Weekly Donchian channel breakouts on 6h timeframe with 1d EMA trend filter and volume confirmation.
# Uses weekly price extremes for structural breakouts, daily EMA for trend alignment, and volume spikes to filter false breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

name = "6h_1w_Donchian_Breakout_TrendFollow_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: highest high over last 20 weekly bars
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 weekly bars
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average for spike detection (24 periods = 4 days of 6h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)  # align daily volume MA to 6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0 * daily average volume
        volume_spike = volume[i] > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with uptrend and volume
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with downtrend and volume
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low or trend changes
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high or trend changes
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals