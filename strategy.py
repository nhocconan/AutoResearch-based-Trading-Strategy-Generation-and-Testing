#!/usr/bin/env python3
# 6h_12h_donchian_weekly_pivot_v1
# Strategy: 6-hour Donchian channel breakout with weekly pivot filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum, weekly pivots provide institutional reference points,
# and volume confirms institutional participation. Works in bull markets via breakout continuation
# and in bear markets via breakdowns with volume, using weekly pivot as trend filter to avoid
# counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Standard pivot point calculation
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 6h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 6h Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above prior period high
        breakdown_down = close[i] < donchian_low[i-1]  # Break below prior period low
        
        # Weekly pivot trend filter
        above_pivot = close[i] > pp_aligned[i]
        below_pivot = close[i] < pp_aligned[i]
        
        # Strong level breakouts (beyond S1/R1)
        strong_breakout_up = breakout_up and close[i] > r1_aligned[i]
        strong_breakdown_down = breakdown_down and close[i] < s1_aligned[i]
        
        # Entry logic: Donchian breakout + volume + pivot alignment
        if strong_breakout_up and vol_confirm[i] and above_pivot and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_breakdown_down and vol_confirm[i] and below_pivot and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian breakout with volume
        elif position == 1 and breakdown_down and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals