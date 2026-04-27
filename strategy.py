#!/usr/bin/env python3
"""
1d_WeeklyPivot_Bias_With_Trend
Hypothesis: Use weekly pivot points (from previous week) to establish bias on daily timeframe.
In uptrend (price > weekly EMA50), buy near weekly S1/S2 with volume confirmation.
In downtrend (price < weekly EMA50), sell near weekly R1/R2 with volume confirmation.
Weekly pivot provides institutional reference points; EMA50 filter ensures trend alignment.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Target: 10-25 trades/year per symbol.
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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_weekly = pd.Series(df_weekly['close'].values)
    ema50_weekly = close_weekly.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate weekly pivot points from previous week
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_pivot = df_weekly['close'].values
    
    # Standard pivot point formula
    pivot = (high_weekly + low_weekly + close_weekly_pivot) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    
    # Align to daily timeframe (previous week's levels available at open)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # Moderate volume filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50  # need 50 for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near support in uptrend with volume spike
            if (close[i] > ema50_weekly_aligned[i] and  # Uptrend filter
                (close[i] <= s1_aligned[i] * 1.02 or close[i] <= s2_aligned[i] * 1.02) and  # Near S1/S2
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near resistance in downtrend with volume spike
            elif (close[i] < ema50_weekly_aligned[i] and  # Downtrend filter
                  (close[i] >= r1_aligned[i] * 0.98 or close[i] >= r2_aligned[i] * 0.98) and  # Near R1/R2
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend fails or reaches pivot/resistance
            if (close[i] < ema50_weekly_aligned[i] or  # Trend fails
                close[i] >= pivot_aligned[i]):  # Reached pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend fails or reaches pivot/support
            if (close[i] > ema50_weekly_aligned[i] or  # Trend fails
                close[i] <= pivot_aligned[i]):  # Reached pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Bias_With_Trend"
timeframe = "1d"
leverage = 1.0