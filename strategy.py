#!/usr/bin/env python3
# 6h_1d_1w_WeeklyPivot_DailyTrend_Confluence
# Hypothesis: On 6h timeframe, trade breakouts from weekly pivot levels (R1/S1) only when aligned with daily trend (EMA50) and confirmed by volume spike.
# Uses weekly pivot for structure, daily EMA50 for trend filter, and volume spike for entry confirmation.
# Designed to work in both bull and bear markets by trading with the daily trend while using weekly levels as support/resistance.
# Targets 15-25 trades per year to avoid fee drag.

name = "6h_1d_1w_WeeklyPivot_DailyTrend_Confluence"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly pivot points (using prior week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation (using prior week)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    
    # Weekly pivot point and ranges
    pivot_1w = typical_price_1w
    range_1w = high_1w - low_1w
    
    # Weekly pivot levels: R1, S1
    s1_1w = pivot_1w - (range_1w * 1.0)
    r1_1w = pivot_1w + (range_1w * 1.0)
    
    # Align weekly levels and daily EMA to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1, volume spike, and price above daily EMA50 (uptrend)
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S1, volume spike, and price below daily EMA50 (downtrend)
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or trend reversal (below EMA50)
            if close[i] < s1_aligned[i] * 0.998 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 or trend reversal (above EMA50)
            if close[i] > r1_aligned[i] * 1.002 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals