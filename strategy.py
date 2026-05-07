#!/usr/bin/env python3
"""
12H_Donchian_Breakout_1D_Trend_Volume
Hypothesis: Use 1d ADX for trend strength and 12h Donchian channel breakout for entry.
Long when price breaks above 12h upper Donchian(20) and 1d ADX > 25;
Short when price breaks below 12h lower Donchian(20) and 1d ADX > 25.
Volume confirmation: current volume > 1.3x 20-period average volume.
This captures strong trend moves with volatility-based breakouts, filtering out weak/choppy markets.
Designed for 12h timeframe to keep trade frequency low (target: 12-37 trades/year).
Works in both bull (breakouts up) and bear (breakouts down) markets via symmetric logic.
"""
name = "12H_Donchian_Breakout_1D_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = (high_1d - close_1d.shift(1)).abs()
    tr3 = (low_1d - close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up = high_1d.diff()
    down = -low_1d.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr.replace(0, 1e-10)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, 1e-10)
    
    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    
    # Donchian channels (20-period)
    upper = high_12h.rolling(window=20, min_periods=20).max()
    lower = low_12h.rolling(window=20, min_periods=20).min()
    upper_values = upper.values
    lower_values = lower.values
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_values)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_values)
    
    # Volume filter: current volume > 1.3x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (2 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price breaks above upper Donchian and strong trend (ADX > 25)
            if (high[i] > upper_12h_aligned[i] and adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below lower Donchian and strong trend (ADX > 25)
            elif (low[i] < lower_12h_aligned[i] and adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion within channel)
            if position == 1 and low[i] < lower_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and high[i] > upper_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals