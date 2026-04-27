#!/usr/bin/env python3
"""
6h_Donchian_20_WeeklyPivot_Direction_VolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (based on 1w high/low) and volume confirmation capture high-probability trend moves. Weekly pivot provides structural bias (above/below weekly pivot = bullish/bearish bias). Volume filter ensures breakouts have conviction. Designed to work in both bull (breakouts with volume) and bear (short breakdowns with volume) markets. Target: 50-150 total trades over 4 years.
"""

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
    
    # Get 1w data for weekly pivot (using prior week's high, low, close)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Weekly bias: above pivot = bullish bias, below pivot = bearish bias
    weekly_bullish = close_1w > weekly_pivot
    weekly_bearish = close_1w < weekly_pivot
    
    # Donchian channel (20-period) - using current bar's high/low for breakout
    # Note: We use rolling window on past 20 bars (excluding current) for breakout level
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align weekly bias and volume confirmation to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)  # volume is LTF, but align using 1w for consistency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level to reduce churn)
    
    # Warmup: need Donchian (20), volume avg (20)
    start_idx = max(20, 20) + 1  # +1 due to shift(1) in Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        weekly_bull = weekly_bullish_aligned[i]
        weekly_bear = weekly_bearish_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Long bias: price breaks above Donchian high with volume and weekly bullish bias
            if weekly_bull and vol_conf and close_val > donch_high:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short bias: price breaks below Donchian low with volume and weekly bearish bias
            elif weekly_bear and vol_conf and close_val < donch_low:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price retouches Donchian low (mean reversion) or weekly bias flips
            if close_val <= donch_low or not weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price retouches Donchian high or weekly bias flips
            if close_val >= donch_high or not weekly_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_Direction_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0