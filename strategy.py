#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyBreakout_TrendFilter
# Hypothesis: Combines weekly pivot context (from 1w data) with daily breakout confirmation and 1d EMA trend filter.
# Weekly pivot defines market structure (bull/bear bias), daily breakout provides entry, 1d EMA ensures trend alignment.
# Works in bull markets by buying dips in uptrend, works in bear markets by selling rallies in downtrend.
# Target: 15-30 trades/year (~60-120 total over 4 years), low frequency to minimize fee drag.

name = "6h_WeeklyPivot_DailyBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot context (trend bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for breakout levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily breakout levels (using prior day's range)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    range_1d = high_1d_prev - low_1d_prev
    breakout_high = close_1d_prev + 0.5 * range_1d  # 50% breakout
    breakout_low = close_1d_prev - 0.5 * range_1d
    
    # Align all indicators to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    breakout_high_6h = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_6h = align_htf_to_ltf(prices, df_1d, breakout_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(breakout_high_6h[i]) or np.isnan(breakout_low_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly pivot (bullish bias) and breaks above daily breakout level with uptrend
            if close[i] > pp_6h[i] and close[i] > breakout_high_6h[i] and close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot (bearish bias) and breaks below daily breakout level with downtrend
            elif close[i] < pp_6h[i] and close[i] < breakout_low_6h[i] and close[i] < ema_34_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below weekly pivot (trend change) or below daily EMA
            if close[i] < pp_6h[i] or close[i] < ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above weekly pivot (trend change) or above daily EMA
            if close[i] > pp_6h[i] or close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals