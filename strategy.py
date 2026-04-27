#!/usr/bin/env python3
"""
6h_WeeklyPivot_Reversion_1dTrend_Filter
Hypothesis: Price reverses from weekly pivot levels (S1/S2/R1/R2) in the direction of daily trend.
Works in bull/bear via trend filter: long only above daily EMA50, short only below.
Targets weekly mean reversion with daily trend filter to avoid counter-trend trades.
Target: 20-30 trades/year on 6h to minimize fee drag.
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
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's OHLC
    wk_high = df_w['high'].values
    wk_low = df_w['low'].values
    wk_close = df_w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (wk_high + wk_low + wk_close) / 3.0
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - wk_high
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - wk_low
    # Support 2 = P - (H - L)
    s2 = pivot - (wk_high - wk_low)
    # Resistance 2 = P + (H - L)
    r2 = pivot + (wk_high - wk_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Daily trend: EMA50
    ema50_d = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and pivot calculation
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_d_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema50_d_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        # Define pivot zones with small buffer
        buffer = 0.001 * price  # 0.1% buffer
        
        if position == 0:
            # Long: price near S1/S2 and above daily EMA (uptrend)
            near_s1 = abs(price - s1_val) <= buffer
            near_s2 = abs(price - s2_val) <= buffer
            in_s_zone = (price >= s2_val - buffer and price <= s1_val + buffer)
            
            if (near_s1 or near_s2 or in_s_zone) and price > ema_trend:
                signals[i] = size
                position = 1
            # Short: price near R1/R2 and below daily EMA (downtrend)
            elif (abs(price - r1_val) <= buffer or abs(price - r2_val) <= buffer or 
                  (price >= r1_val - buffer and price <= r2_val + buffer)) and price < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses pivot or trend turns down
            if price >= pivot_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses pivot or trend turns up
            if price <= pivot_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Reversion_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0