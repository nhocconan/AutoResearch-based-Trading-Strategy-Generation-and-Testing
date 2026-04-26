#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_VolumeSpike
Hypothesis: On 6h timeframe, use Donchian(20) breakouts filtered by weekly pivot direction and volume confirmation. Enter long when price breaks above Donchian upper band with price above weekly pivot (bullish bias) and volume spike. Enter short when price breaks below Donchian lower band with price below weekly pivot (bearish bias) and volume spike. Weekly pivot provides structural bias from higher timeframe, reducing false breakouts in choppy markets. Designed for 12-30 trades/year on 6h by requiring weekly alignment and volume confirmation, reducing overtrading while capturing structured moves.
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    prev_1w_high = df_1w['high'].shift(1).values
    prev_1w_low = df_1w['low'].shift(1).values
    prev_1w_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_1w_high + prev_1w_low + prev_1w_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup, volume MA warmup, weekly pivot alignment
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly pivot bias
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + price above weekly pivot + volume spike
            long_signal = (close[i] > donchian_upper[i]) and price_above_pivot and volume_spike[i]
            
            # Short: price breaks below Donchian lower + price below weekly pivot + volume spike
            short_signal = (close[i] < donchian_lower[i]) and price_below_pivot and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower OR price crosses below weekly pivot
            if (close[i] < donchian_lower[i] or not price_above_pivot):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR price crosses above weekly pivot
            if (close[i] > donchian_upper[i] or not price_below_pivot):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0