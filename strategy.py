#!/usr/bin/env python3
"""
4h_Donchian20_VolumeTrend_12hEMA50_v1
Hypothesis: 4h Donchian(20) breakout in direction of 12h EMA50 trend with volume confirmation.
Donchian channels provide objective volatility-based support/resistance levels.
12h EMA50 trend filter ensures we only take breakouts aligned with intermediate trend.
Volume confirmation adds conviction filter. Discrete sizing (0.25) limits fee drag.
Target: 75-200 total trades over 4 years (19-50/year) by requiring HTF alignment, Donchian breakout, trend alignment, and volume.
Works in both bull and bear markets due to trend filter and volume confirmation reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF EMA
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_series_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (completed 12h bars only)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian(20) channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_above = close[i] > donchian_upper[i]  # Break above upper channel
        breakout_below = close[i] < donchian_lower[i]   # Break below lower channel
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: Donchian upper breakout with volume, in 12h uptrend
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_below and volume_spike and downtrend:
            # Short signal: Donchian lower breakout with volume, in 12h downtrend
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeTrend_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0