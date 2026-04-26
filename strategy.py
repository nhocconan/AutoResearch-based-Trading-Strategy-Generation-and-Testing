#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_VolumeConfirm
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w HTF) with volume confirmation.
Weekly pivot provides structural bias (bullish/bearish/neutral) from higher timeframe, reducing false breakouts in ranging markets.
Donchian(20) captures intermediate-term breakouts. Volume confirmation ensures breakout validity.
Designed for 6h timeframe to achieve 12-37 trades/year (50-150 over 4 years) with discrete size 0.25 to limit fee drag.
Works in bull/bear via weekly pivot filter + volume confirmation reducing false signals.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on prior week OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_range = high_1w - low_1w
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + weekly_range
    s2 = pivot - weekly_range
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly pivot levels to 6h timeframe (prior week's levels available at week start)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for Donchian(20) calculation (using 1d HTF for intermediate structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: volume > 1.8x 30-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (30), aligned indicators
    start_idx = max(30, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine weekly pivot bias
        # Bullish bias: price above weekly pivot
        # Bearish bias: price below weekly pivot
        # Neutral: near pivot (within 0.5% of pivot) - no new entries
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        near_pivot = abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.005
        
        if position == 0:
            # Long: price breaks above Donchian upper + bullish bias + volume spike
            long_breakout = close[i] > donchian_upper_aligned[i]
            long_signal = long_breakout and bullish_bias and volume_spike[i] and not near_pivot
            
            # Short: price breaks below Donchian lower + bearish bias + volume spike
            short_breakout = close[i] < donchian_lower_aligned[i]
            short_signal = short_breakout and bearish_bias and volume_spike[i] and not near_pivot
            
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
            # Exit: price touches Donchian lower OR weekly bias turns bearish (price below pivot)
            if (close[i] < donchian_lower_aligned[i] or not bullish_bias):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian upper OR weekly bias turns bullish (price above pivot)
            if (close[i] > donchian_upper_aligned[i] or not bearish_bias):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0