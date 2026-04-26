#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dEMA50_Trend_VolumeSpike
Hypothesis: Weekly Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Weekly structure provides strong support/resistance levels that work in both bull and bear markets.
6h timeframe allows for timely execution while avoiding excessive fee drag. Designed for low trade frequency (12-37/year).
"""

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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Upper and lower bands
    upper_band = high_roll
    lower_band = low_roll
    
    # Align weekly Donchian to 6h
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly Donchian (20), daily EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_50_val = ema_50_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly upper band with volume confirmation and uptrend
            long_signal = (close_val > upper_val) and (volume_val > 1.8 * vol_ma_val) and (close_val > ema_50_val)
            # Short: price breaks below weekly lower band with volume confirmation and downtrend
            short_signal = (close_val < lower_val) and (volume_val > 1.8 * vol_ma_val) and (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly lower band or trend reversal
            if (close_val < lower_val or 
                close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly upper band or trend reversal
            if (close_val > upper_val or 
                close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0