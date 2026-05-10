#/usr/bin/env python3
# 12h_Donchian_20_1dTrend_WeeklyPivot_Filter
# Hypothesis: Breakouts from Donchian(20) channels on 12h with 1d trend filter (EMA50) and weekly pivot support/resistance as confluence.
# Uses 12h timeframe to reduce trade frequency, targets 15-35 trades/year. Works in bull/bear via trend alignment.
# Weekly pivot adds institutional level confluence to filter false breakouts.

name = "12h_Donchian_20_1dTrend_WeeklyPivot_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Weekly pivot points from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian channel (20-period) on 12h data
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= donchian_period - 1:
            start_idx = i - donchian_period + 1
            upper[i] = np.max(high[start_idx:i+1])
            lower[i] = np.min(low[start_idx:i+1])
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_period)  # Need enough data
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume, 1d uptrend, and above weekly S1
            if (high[i] > upper[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirm[i] and
                close[i] > s1_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume, 1d downtrend, and below weekly R1
            elif (low[i] < lower[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirm[i] and
                  close[i] < r1_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian lower or 1d trend turns down or below weekly pivot
            if (low[i] < lower[i] or
                trend_1d_up_aligned[i] < 0.5 or
                close[i] < pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian upper or 1d trend turns up or above weekly pivot
            if (high[i] > upper[i] or
                trend_1d_down_aligned[i] < 0.5 or
                close[i] > pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals