#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend
Hypothesis: On 6h timeframe, price breaking above/below 20-bar Donchian channel with alignment to weekly pivot bias and 1d EMA50 trend filter. 
Weekly pivot provides structural support/resistance from higher timeframe, Donchian breakout captures momentum, and 1d EMA50 ensures trend alignment.
This combination should work in both bull and bear markets by only taking breakouts in direction of higher timeframe trend.
Designed for low trade frequency (~20-40/year) to minimize fee drag while maintaining edge through confluence of HTF structure, momentum breakout, and trend filter.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot points (requires full week OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align HTF indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w, additional_delay_bars=1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w, additional_delay_bars=1)
    
    # Calculate 20-bar Donchian channel on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and Donchian (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter and weekly pivot bias
            # Long: price breaks above Donchian high AND above weekly pivot AND above EMA50 (uptrend)
            # Short: price breaks below Donchian low AND below weekly pivot AND below EMA50 (downtrend)
            long_signal = (close[i] > highest_20[i]) and (close[i] > pivot_aligned[i]) and (close[i] > ema50_aligned[i])
            short_signal = (close[i] < lowest_20[i]) and (close[i] < pivot_aligned[i]) and (close[i] < ema50_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below weekly pivot (mean reversion to pivot)
            exit_signal = close[i] < pivot_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above weekly pivot (mean reversion to pivot)
            exit_signal = close[i] > pivot_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0