#!/usr/bin/env python3
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume filter: current volume > 1.5 * 10-period average (10 periods = 120h = 5 days)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # 12h EMA crossover for trend filter
    close_series = pd.Series(close)
    ema9_12h = close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_12h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 21)  # Need sufficient data for volume MA and EMA21
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(volume_ma10[i]) or
            np.isnan(ema9_12h[i]) or np.isnan(ema21_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        # Trend filter: EMA9 > EMA21 for uptrend, EMA9 < EMA21 for downtrend
        uptrend = ema9_12h[i] > ema21_12h[i]
        downtrend = ema9_12h[i] < ema21_12h[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume, in uptrend
            if (close[i] > r1_12h[i] and volume_filter and uptrend):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume, in downtrend
            elif (close[i] < s1_12h[i] and volume_filter and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S2 (deeper level) or trend breaks
            if close[i] < s2_12h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R2 or trend breaks
            if close[i] > r2_12h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0