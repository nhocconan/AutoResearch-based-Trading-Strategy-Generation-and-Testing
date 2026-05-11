#!/usr/bin/env python3
"""
1d_1W_TF_WeeklyTrend_DailyBreakout
Hypothesis: Trade daily breakouts at prior day's high/low aligned with weekly trend.
In bull markets, buy breakouts above prior day's high when weekly trend is up.
In bear markets, sell breakdowns below prior day's low when weekly trend is down.
Uses 1d timeframe with 1w trend filter to reduce false signals and capture directional moves.
Target: 15-25 trades/year (60-100 total over 4 years) by requiring weekly trend alignment.
"""

name = "1d_1W_TF_WeeklyTrend_DailyBreakout"
timeframe = "1d"
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
    
    # === Weekly Trend Filter (EMA34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily Breakout Levels (prior day high/low) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph = df_1d['high'].values  # prior day high
    pl = df_1d['low'].values   # prior day low
    
    # Align prior day levels to 1d timeframe (already aligned by get_htf_data)
    # get_htf_data returns the actual OHLC for each 1d bar, so we can use directly
    # but we need to shift by 1 to get prior day's levels
    ph_shifted = np.roll(ph, 1)
    pl_shifted = np.roll(pl, 1)
    ph_shifted[0] = np.nan  # first day has no prior day
    pl_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 34 periods for weekly EMA and 1 for shift)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d[i]) or np.isnan(ph_shifted[i]) or np.isnan(pl_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above prior day's high with weekly uptrend
            if (close[i] > ph_shifted[i] and close[i] > ema34_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below prior day's low with weekly downtrend
            elif (close[i] < pl_shifted[i] and close[i] < ema34_1d[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below prior day's low (reversal)
            if close[i] < pl_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above prior day's high (reversal)
            if close[i] > ph_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals