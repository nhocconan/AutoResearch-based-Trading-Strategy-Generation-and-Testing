#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Breakout
# Hypothesis: Weekly pivot levels (from weekly OHLC) act as strong support/resistance.
# Breakouts above weekly R2 or below weekly S2 with daily trend alignment and volume
# confirmation capture strong momentum moves. Works in bull (breakouts) and bear (mean
# reversion at extremes) with tight entries to avoid overtrading.

name = "6h_WeeklyPivot_DailyTrend_Breakout"
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
    
    # 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d data for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly pivot levels (standard calculation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    R1 = pivot_1w + range_1w
    S1 = pivot_1w - range_1w
    R2 = pivot_1w + 2 * range_1w
    S2 = pivot_1w - 2 * range_1w
    
    # Align weekly pivot levels to 6h
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Daily EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above weekly R2 with daily uptrend and volume spike
            if (close[i] > R2_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S2 with daily downtrend and volume spike
            elif (close[i] < S2_aligned[i] and 
                  trend_1d_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below weekly S1 or daily trend fails
            S1 = pivot_1w[i] - range_1w[i] if not (np.isnan(pivot_1w[i]) or np.isnan(range_1w[i])) else np.nan
            S1_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, S1))[i] if not np.isnan(S1) else np.nan
            if (close[i] < S1_aligned or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above weekly R1 or daily trend fails
            R1 = pivot_1w[i] + range_1w[i] if not (np.isnan(pivot_1w[i]) or np.isnan(range_1w[i])) else np.nan
            R1_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, R1))[i] if not np.isnan(R1) else np.nan
            if (close[i] > R1_aligned or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals