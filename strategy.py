#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_trend
# Hypothesis: 6-hour breakout from weekly pivot levels (R2/S2) with 1-day EMA200 trend filter.
# Weekly pivots provide strong institutional levels; EMA200 filter avoids counter-trend trades.
# Works in bull/bear by only trading in direction of higher timeframe trend.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "6h_1d_weekly_pivot_breakout_trend"
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
    
    # Get daily data for weekly pivot and EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous week's high, low, close (using 5-day approximation for weekly)
    # For simplicity, using 5 trading days as proxy for week
    prev_week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    prev_week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    prev_week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(1).values
    
    # Weekly pivot point and support/resistance levels
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pp - prev_week_low
    r2 = pp + (prev_week_high - prev_week_low)
    s1 = 2 * pp - prev_week_high
    s2 = pp - (prev_week_high - prev_week_low)
    
    # EMA200 for trend filter
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly pivot levels and EMA200 to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R2 with uptrend filter (price > EMA200)
        if (close[i] > r2_aligned[i] and close[i] > ema200_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S2 with downtrend filter (price < EMA200)
        elif (close[i] < s2_aligned[i] and close[i] < ema200_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite pivot level
        elif position == 1 and close[i] < pp:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pp:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals