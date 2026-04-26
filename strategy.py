#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: On 6h timeframe, use Donchian(20) breakouts filtered by weekly pivot direction (from 1w timeframe) and volume confirmation (>1.5x 20-period average). Enter long when price breaks above Donchian upper band with weekly bullish bias (close > weekly pivot) and volume spike. Enter short when price breaks below Donchian lower band with weekly bearish bias (close < weekly pivot) and volume spike. Uses discrete position size 0.25 to manage drawdown. Designed for 12-37 trades/year on 6h by requiring weekly alignment and volume confirmation, reducing overtrading while capturing structured moves in both bull and bear markets.
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
    
    # Get 1w data for weekly pivot and bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor pivot)
    # Pivot = (H + L + C) / 3
    # Bias: close > Pivot = bullish, close < Pivot = bearish
    prev_1w_high = df_1w['high'].shift(1).values
    prev_1w_low = df_1w['low'].shift(1).values
    prev_1w_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_1w_high + prev_1w_low + prev_1w_close) / 3.0
    weekly_bias_bullish = prev_1w_close > weekly_pivot
    weekly_bias_bearish = prev_1w_close < weekly_pivot
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish.astype(float))
    
    # Calculate Donchian(20) channels on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian and volume MA warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_bias_bullish_aligned[i]) or np.isnan(weekly_bias_bearish_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly bias alignment (convert back to boolean)
        weekly_bullish = weekly_bias_bullish_aligned[i] > 0.5
        weekly_bearish = weekly_bias_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: price breaks above Donchian upper + weekly bullish bias + volume spike
            long_signal = (close[i] > donchian_upper[i]) and weekly_bullish and volume_spike[i]
            
            # Short: price breaks below Donchian lower + weekly bearish bias + volume spike
            short_signal = (close[i] < donchian_lower[i]) and weekly_bearish and volume_spike[i]
            
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
            # Exit: price breaks below Donchian lower OR weekly bias turns bearish
            if (close[i] < donchian_lower[i] or not weekly_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR weekly bias turns bullish
            if (close[i] > donchian_upper[i] or not weekly_bearish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0