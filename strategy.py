#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian20_Breakout_TrendFilter_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation. 
Weekly pivot provides macro trend bias (R1/S1 from prior week), Donchian captures breakouts in that direction.
Volume > 1.5x 20-period MA confirms breakout strength. Designed for low overtrading (<25 trades/year) 
by requiring confluence of weekly bias, breakout, and volume. Works in bull/bear via weekly trend alignment.
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
    
    # Load weekly data ONCE before loop for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week OHLC)
    # Prior week values (shifted by 1 to avoid look-ahead)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_open = df_1w['open'].shift(1).values
    
    # Weekly pivot point (PP) and support/resistance levels
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly trend: 1 = bullish (close above PP), -1 = bearish (close below PP)
    weekly_trend = np.where(pp_aligned > 0, 
                            np.where(prev_week_close > pp_aligned, 1, -1), 
                            0)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Donchian channel (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * volume_ma(20) for breakout confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with weekly trend filter
        if position == 0:
            # Long: Price breaks above Donchian upper AND weekly bullish trend AND volume spike
            if close[i] > highest_20[i] and weekly_trend_aligned[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND weekly bearish trend AND volume spike
            elif close[i] < lowest_20[i] and weekly_trend_aligned[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian lower OR weekly trend turns bearish
            if close[i] < lowest_20[i] or weekly_trend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian upper OR weekly trend turns bullish
            if close[i] > highest_20[i] or weekly_trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_Donchian20_Breakout_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0