#!/usr/bin/env python3
# Hypothesis: 6h timeframe with daily pivot point (PP) and 1-week trend filter.
# Uses daily pivot point (PP) and support/resistance levels (S1, R1) for mean-reversion entries.
# Weekly EMA34 acts as trend filter: only take longs when price > weekly EMA34, shorts when price < weekly EMA34.
# This combines mean-reversion at key daily levels with trend alignment to avoid counter-trend whipsaws.
# Works in bull markets (trend-following bias) and bear markets (mean-reversion at resistance/support).
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_PivotPoint_1wEMA34_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate daily pivot point and support/resistance from previous day
    prev_high = np.roll(high, 4)   # 4 bars = 1 day (6h * 4 = 24h)
    prev_low = np.roll(low, 4)
    prev_close = np.roll(close, 4)
    prev_high[:4] = np.nan
    prev_low[:4] = np.nan
    prev_close[:4] = np.nan
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    s1 = 2 * pivot_point - prev_high
    r1 = 2 * pivot_point - prev_low
    
    # Mean-reversion signals: price touching S1 (long) or R1 (short)
    touch_s1 = low <= s1
    touch_r1 = high >= r1
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1-week EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Trend filter: only go long in uptrend, short in downtrend
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Volume filter: avoid low-volume false signals
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(touch_s1[i]) or np.isnan(touch_r1[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S1 + weekly uptrend + volume filter
            if touch_s1[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 + weekly downtrend + volume filter
            elif touch_r1[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot point or trend reverses
            if close[i] >= pivot_point[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot point or trend reverses
            if close[i] <= pivot_point[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals