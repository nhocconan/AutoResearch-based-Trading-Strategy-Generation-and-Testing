#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_HTFVolume_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1w) and confirmed by 12h volume spikes produce fewer but higher-quality trades. Weekly pivot provides structural bias (bull/bear) that works in both trending and ranging markets, while volume confirmation avoids false breakouts. Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Load weekly data ONCE before loop for pivot direction (HTF bias)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    # Bias: above P = bullish, below P = bearish
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_bullish = close_1w > weekly_pivot  # bullish bias
    weekly_bearish = close_1w < weekly_pivot  # bearish bias
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h volume average for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # 6h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation (12h volume > 1.5x 20-period average)
        volume_spike = volume[i] > 1.5 * vol_ma_20_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high_20[i]
        breakout_low = close[i] < donchian_low_20[i]
        
        # Long logic: breakout above Donchian high with weekly bullish bias and volume
        if breakout_high and weekly_bullish_aligned[i] > 0.5 and volume_spike:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below Donchian low with weekly bearish bias and volume
        elif breakout_low and weekly_bearish_aligned[i] > 0.5 and volume_spike:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite Donchian breakout OR loss of volume confirmation
        elif position == 1 and (breakout_low or not volume_spike):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_high or not volume_spike):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_HTFVolume_v1"
timeframe = "6h"
leverage = 1.0