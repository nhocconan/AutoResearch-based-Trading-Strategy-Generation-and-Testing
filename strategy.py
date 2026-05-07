#!/usr/bin/env python3
# 6H_WeeklyPivot_Breakout_1dTrend_Volume
# Hypothesis: Breakouts above weekly pivot R4 or below S4 with 1d trend filter and volume spike.
# Weekly pivots provide strong support/resistance; breakouts capture momentum in both bull/bear markets.
# Volume confirmation reduces false signals. Target: 20-50 trades/year.

name = "6H_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly R4 and S4 levels (aggressive breakout levels)
    r4_1w = pivot_1w + 3 * range_1w
    s4_1w = pivot_1w - 3 * range_1w
    
    # Align weekly levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R4, price above daily EMA34 (uptrend), volume spike (>2x)
            if (close[i] > r4_1w_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4, price below daily EMA34 (downtrend), volume spike (>2x)
            elif (close[i] < s4_1w_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below weekly pivot (mean reversion)
            if close[i] <= pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above weekly pivot (mean reversion)
            if close[i] >= pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals