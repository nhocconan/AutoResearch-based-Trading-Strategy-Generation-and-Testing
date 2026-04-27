#!/usr/bin/env python3
"""
#100895 - 6h_Pivots_Trend_Follow_1wEMA_Trend_Filter
Hypothesis: Daily pivot-based trend following with weekly EMA50 filter on 6h timeframe. 
Trades only when price is above/below daily pivot (mean reversion avoided) and aligned with weekly trend.
Targets 12-37 trades/year (50-150 total) to minimize fee drag. Uses 6m price action with 
daily pivots and weekly EMA for trend filter. Works in bull (buy dips above pivot) and bear 
(sell rallies below pivot) by following weekly trend.
"""

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Daily pivot calculation
    daily_pivot = (prev_high + prev_low + prev_close) / 3
    daily_r1 = 2 * daily_pivot - prev_low
    daily_s1 = 2 * daily_pivot - prev_high
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above pivot and above weekly EMA50 with volume
        if (close[i] > pivot_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below pivot and below weekly EMA50 with volume
        elif (close[i] < pivot_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses back below/above weekly EMA (trend change)
        elif position == 1 and close[i] < ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Pivots_Trend_Follow_1wEMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0