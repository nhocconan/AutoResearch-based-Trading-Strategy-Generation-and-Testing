#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dEMA34_Filter
Hypothesis: Weekly pivot points (R1, S1) act as significant support/resistance.
Breakouts above weekly R1 or below weekly S1 with 1d EMA34 trend filter capture
institutional moves. Works in bull (buy R1 breaks above EMA34 uptrend) and bear
(sell S1 breaks below EMA34 downtrend). Uses 6h timeframe for lower frequency
to reduce fee drag. Target: 15-25 trades/year.
"""

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
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points (R1, S1) from prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Prior week's values (shifted by 1 to avoid look-ahead)
    phigh = np.roll(high_1w, 1)
    plow = np.roll(low_1w, 1)
    pclose = np.roll(close_1w, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Weekly pivot point and support/resistance levels
    pp = (phigh + plow + pclose) / 3.0
    r1 = 2 * pp - plow
    s1 = 2 * pp - phigh
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and pivots
    start_idx = 34  # for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: break above weekly R1 with price above EMA34 (uptrend)
            if close[i] > r1_val and close[i] > ema_34_val:
                signals[i] = size
                position = 1
            # Short: break below weekly S1 with price below EMA34 (downtrend)
            elif close[i] < s1_val and close[i] < ema_34_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly R1 (failed breakout) or below EMA34 (trend change)
            if close[i] < r1_val or close[i] < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above weekly S1 (failed breakdown) or above EMA34 (trend change)
            if close[i] > s1_val or close[i] > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Direction_1dEMA34_Filter"
timeframe = "6h"
leverage = 1.0