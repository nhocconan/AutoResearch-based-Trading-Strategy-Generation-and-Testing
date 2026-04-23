#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d Weekly Pivot Confluence
- Williams %R(14) on 6h: Long when < -90 (oversold), Short when > -10 (overbought)
- 1d Weekly Pivot: Weekly PP, R1, S1 from prior week's OHLC
- Long: Williams %R < -90 AND price > Weekly PP AND price < Weekly R1 (mean reversion in uptrend)
- Short: Williams %R > -10 AND price < Weekly PP AND price > Weekly S1 (mean reversion in downtrend)
- Exit: Williams %R returns to > -50 (for long) or < -50 (for short) OR weekly pivot flip
- Uses extreme momentum exhaustion for mean reversion entries, filtered by weekly pivot structure
- Target: 75-150 total trades over 4 years (19-38/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
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
    
    # Williams %R(14) on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 1d Weekly Pivot from prior week's OHLC
    df_1d = get_htf_data(prices, '1d')
    # Resample 1d to weekly using actual weekly boundaries (no look-ahead)
    # We'll compute weekly pivot using prior week's data for each 1d bar
    # For simplicity, we use prior 5 trading days (1 week) OHLC
    window = 5  # 5 trading days = 1 week
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=window, min_periods=window).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=window, min_periods=window).mean().values  # approx weekly close
    
    # Weekly Pivot: PP = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly R1 = 2*PP - L
    weekly_r1 = 2 * weekly_pp - weekly_low
    # Weekly S1 = 2*PP - H
    weekly_s1 = 2 * weekly_pp - weekly_high
    
    # Align weekly pivot to 6h timeframe (using prior week's data)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 5)  # Need 14 for Williams %R, 5 for weekly pivot
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(weekly_pp_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -90 (oversold) AND price > Weekly PP AND price < Weekly R1
            if (williams_r[i] < -90 and 
                close[i] > weekly_pp_aligned[i] and 
                close[i] < weekly_r1_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (overbought) AND price < Weekly PP AND price > Weekly S1
            elif (williams_r[i] > -10 and 
                  close[i] < weekly_pp_aligned[i] and 
                  close[i] > weekly_s1_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to > -50 (momentum returning) OR weekly pivot flip
            if williams_r[i] > -50 or close[i] < weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to < -50 (momentum returning) OR weekly pivot flip
            if williams_r[i] < -50 or close[i] > weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_WeeklyPivot_MeanReversion"
timeframe = "6h"
leverage = 1.0