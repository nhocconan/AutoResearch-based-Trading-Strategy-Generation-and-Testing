#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeReversion
Hypothesis: Weekly pivot levels act as strong support/resistance. Price tends to reverse
from weekly S1/R1 in ranging markets (identified by weekly ADX < 25). Works in both
bull and bear by fading extremes when range-bound, avoiding trends.
Target: 15-30 trades/year on 6h to minimize fee drag.
"""

name = "6h_WeeklyPivot_RangeReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points and ADX (trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot points (standard formula)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Weekly ADX for trend/ranging filter (ADX < 25 = ranging)
    # Calculate ADX components
    plus_dm = np.zeros_like(high_w)
    minus_dm = np.zeros_like(low_w)
    tr = np.zeros_like(high_w)
    
    for i in range(1, len(high_w)):
        plus_dm[i] = max(high_w[i] - high_w[i-1], 0) if high_w[i] - high_w[i-1] > high_w[i-1] - low_w[i] else 0
        minus_dm[i] = max(low_w[i-1] - low_w[i], 0) if low_w[i-1] - low_w[i] > high_w[i] - high_w[i-1] else 0
        tr[i] = max(high_w[i] - low_w[i], abs(high_w[i] - high_w[i-1]), abs(low_w[i] - low_w[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr_w = np.zeros_like(tr)
    plus_di_w = np.zeros_like(plus_dm)
    minus_di_w = np.zeros_like(minus_dm)
    
    # Initial values
    atr_w[period] = np.mean(tr[1:period+1])
    plus_di_w[period] = np.mean(plus_dm[1:period+1]) / atr_w[period] * 100 if atr_w[period] != 0 else 0
    minus_di_w[period] = np.mean(minus_dm[1:period+1]) / atr_w[period] * 100 if atr_w[period] != 0 else 0
    
    # Wilder smoothing
    for i in range(period+1, len(tr)):
        atr_w[i] = (atr_w[i-1] * (period-1) + tr[i]) / period
        plus_di_w[i] = (plus_di_w[i-1] * (period-1) + plus_dm[i]) / period / atr_w[i] * 100 if atr_w[i] != 0 else 0
        minus_di_w[i] = (minus_di_w[i-1] * (period-1) + minus_dm[i]) / period / atr_w[i] * 100 if atr_w[i] != 0 else 0
    
    # DX and ADX
    dx_w = np.zeros_like(tr)
    adx_w = np.zeros_like(tr)
    
    for i in range(period*2, len(tr)):
        di_sum = plus_di_w[i] + minus_di_w[i]
        if di_sum != 0:
            dx_w[i] = abs(plus_di_w[i] - minus_di_w[i]) / di_sum * 100
        else:
            dx_w[i] = 0
    
    # Smooth DX to get ADX
    adx_start = period*2
    if len(tr) > adx_start + period:
        adx_w[adx_start] = np.mean(dx_w[adx_start+1:adx_start+1+period])
        for i in range(adx_start+1, len(tr)):
            adx_w[i] = (adx_w[i-1] * (period-1) + dx_w[i]) / period
    
    # Align weekly ADX to 6h timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # Get price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough weekly data for ADX
    start_idx = period * 3  # Ensure ADX is stable
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_w_aligned[i]) or
            np.isnan(adx_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in ranging markets (weekly ADX < 25)
        if adx_w_aligned[i] >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or goes below S1, then reverses back above S1
            # Wait for confirmation: close above S1 after touching below
            if low[i] <= s1_w_aligned[i] and close[i] > s1_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R1, then reverses back below R1
            elif high[i] >= r1_w_aligned[i] and close[i] < r1_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or S2, or ADX starts trending
            if close[i] >= pivot_w_aligned[i] or close[i] <= s2_w_aligned[i] or adx_w_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or R2, or ADX starts trending
            if close[i] <= pivot_w_aligned[i] or close[i] >= r2_w_aligned[i] or adx_w_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals