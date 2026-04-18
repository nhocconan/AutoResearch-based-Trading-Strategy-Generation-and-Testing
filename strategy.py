# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_WeeklyPivotDirection_DonchianBreakout_VolumeFilter
Hypothesis: Combine weekly pivot trend direction (from 1w) with Donchian(20) breakout and volume confirmation on 6h.
In bull markets (price > weekly pivot), go long on upper band breakouts with volume.
In bear markets (price < weekly pivot), go short on lower band breakouts with volume.
This captures breakout moves aligned with higher timeframe trend, reducing whipsaw in sideways markets.
Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: P = (H + L + C) / 3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_pivot = calculate_weekly_pivot(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 6d data for Donchian channels (20-period high/low)
    df_6d = get_htf_data(prices, '6d')  # ~20 periods of 6h = ~5 days, use 6d for stability
    if len(df_6d) == 0:
        # Fallback to 1d if 6d not available
        df_6d = get_htf_data(prices, '1d')
        if len(df_6d) == 0:
            return np.zeros(n)
    
    # Donchian(20) on 6d/1d
    donchian_high = pd.Series(df_6d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_6d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6d, donchian_low)
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup periods
    start_idx = max(20, 20)  # Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_confirm = volume_confirm[i]
        
        # Determine market regime from weekly pivot
        is_bullish = price > weekly_pivot_val
        is_bearish = price < weekly_pivot_val
        
        if position == 0:
            # Long: bullish regime + price breaks above upper band + volume confirmation
            if is_bullish and price > upper_band and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish regime + price breaks below lower band + volume confirmation
            elif is_bearish and price < lower_band and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long position
            signals[i] = 0.25
            # Exit: price breaks below lower band (reversal) OR bearish regime shift
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            elif not is_bullish:  # regime changed to bearish
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Maintain short position
            signals[i] = -0.25
            # Exit: price breaks above upper band (reversal) OR bullish regime shift
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            elif not is_bearish:  # regime changed to bullish
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivotDirection_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0