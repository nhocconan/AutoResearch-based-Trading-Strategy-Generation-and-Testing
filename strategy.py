# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume
1d Donchian breakout with 1w trend filter and volume confirmation.
Enters long when price breaks above 20-day Donchian high with 1w bullish trend (price > 50-week SMA).
Enters short when price breaks below 20-day Donchian low with 1w bearish trend (price < 50-week SMA).
Requires volume > 1.5x 20-day average.
Exits when price crosses 10-day SMA in opposite direction.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
Works in trending markets by following higher timeframe trend.
"""

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get daily data for Donchian and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 20-day Donchian channels
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 10-day SMA for exit
    sma_10 = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (vol_ma * 1.5)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 50-week SMA for trend filter
    sma_50w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    sma_10_aligned = align_htf_to_ltf(prices, df_1d, sma_10)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    sma_50w_aligned = align_htf_to_ltf(prices, df_1w, sma_50w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(sma_10_aligned[i]) or np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(sma_50w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian high with 1w bullish trend and volume
        if (close[i] > donch_high_aligned[i] and close[i] > sma_50w_aligned[i] and 
            vol_confirm_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below Donchian low with 1w bearish trend and volume
        elif (close[i] < donch_low_aligned[i] and close[i] < sma_50w_aligned[i] and 
              vol_confirm_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses 10-day SMA in opposite direction
        elif position == 1 and close[i] < sma_10_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > sma_10_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals