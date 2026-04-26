#!/usr/bin/env python3
"""
6h_WeeklyPivot_PB_Donchian20_1dEMA50_Trend
Hypothesis: On 6h timeframe, enter long when price pulls back to weekly pivot support (S1/S2) during uptrend (1d EMA50) and breaks above prior 20-bar high with volume confirmation. Enter short when price pulls back to weekly pivot resistance (R1/R2) during downtrend and breaks below prior 20-bar low with volume confirmation. Uses weekly pivot for structure, 1d EMA50 for trend filter, and Donchian breakout for entry timing to reduce false signals. Designed to work in both bull and bear markets by following the higher timeframe trend.
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    r1 = 2 * pivot - prev_low_1w
    s1 = 2 * pivot - prev_high_1w
    r2 = pivot + range_1w
    s2 = pivot - range_1w
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    # Donchian channels for breakout confirmation
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and Donchian warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Pullback to pivot support/resistance conditions
        pullback_to_support = (low[i] <= s1_aligned[i] * 1.005) or (low[i] <= s2_aligned[i] * 1.005)
        pullback_to_resistance = (high[i] >= r1_aligned[i] * 0.995) or (high[i] >= r2_aligned[i] * 0.995)
        
        # Donchian breakout conditions
        breakout_above = high[i] > donchian_high[i]
        breakout_below = low[i] < donchian_low[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: pullback to weekly support + volume spike + Donchian breakout above + 1d uptrend
            long_signal = pullback_to_support and volume_spike[i] and breakout_above and trend_uptrend
            
            # Short: pullback to weekly resistance + volume spike + Donchian breakout below + 1d downtrend
            short_signal = pullback_to_resistance and volume_spike[i] and breakout_below and trend_downtrend
            
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
            # Exit: trend change to downtrend OR price breaks below weekly S2
            if not trend_uptrend or low[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change to uptrend OR price breaks above weekly R2
            if not trend_downtrend or high[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_PB_Donchian20_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0