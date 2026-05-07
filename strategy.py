#!/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_1dTrend_Volume
Hypothesis: Weekly pivot levels act as dynamic support/resistance. In 6h timeframe,
we look for pullbacks to weekly pivot (PP) or S1/R1 with confirmation from daily trend
and volume. Works in bull/bear via daily trend filter. Targets 12-35 trades/year.
"""

name = "6h_WeeklyPivot_Pullback_1dTrend_Volume"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # Weekly pivot from previous week (high, low, close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous week's data (last completed week)
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Calculate pivot and levels for the current week (based on previous week)
    pivot = (wh + wl + wc) / 3.0
    r1 = 2 * pivot - wl
    s1 = 2 * pivot - wh
    r2 = pivot + (wh - wl)
    s2 = pivot - (wh - wl)
    
    # Align weekly levels to 6h timeframe (values change only when new week starts)
    pivot_64 = align_htf_to_ltf(prices, df_1w, pivot)
    r1_64 = align_htf_to_ltf(prices, df_1w, r1)
    s1_64 = align_htf_to_ltf(prices, df_1w, s1)
    r2_64 = align_htf_to_ltf(prices, df_1w, r2)
    s2_64 = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_64[i]) or np.isnan(r1_64[i]) or np.isnan(s1_64[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to S1 or S2 with daily uptrend and volume
            if (close[i] <= s1_64[i] * 1.005 or close[i] <= s2_64[i] * 1.005) and \
               close[i] > ema_1d_aligned[i] and \
               volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: pullback to R1 or R2 with daily downtrend and volume
            elif (close[i] >= r1_64[i] * 0.995 or close[i] >= r2_64[i] * 0.995) and \
                 close[i] < ema_1d_aligned[i] and \
                 volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches pivot or R1 (mean reversion to weekly pivot)
            if close[i] >= pivot_64[i] * 0.995 or close[i] >= r1_64[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches pivot or S1 (mean reversion to weekly pivot)
            if close[i] <= pivot_64[i] * 1.005 or close[i] <= s1_64[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals