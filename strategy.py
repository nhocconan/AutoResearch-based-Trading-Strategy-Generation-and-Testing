#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Weekly pivot points (calculated from prior week OHLC) provide strong support/resistance.
# Breakouts above R1 or below S1 with 1d EMA34 trend filter and volume confirmation capture institutional moves.
# Weekly timeframe reduces noise, suitable for 6h to achieve 12-37 trades/year. Works in bull/bear via trend filter.

name = "6h_Weekly_Pivot_Breakout_1dTrend_Volume"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (based on prior week)
    def calculate_pivot_point(h, l, c):
        pivot = (h + l + c) / 3.0
        R1 = 2 * pivot - l
        S1 = 2 * pivot - h
        return R1, S1
    
    R1 = np.full_like(close_1w, np.nan)
    S1 = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        R1[i], S1[i] = calculate_pivot_point(high_1w[i-1], low_1w[i-1], close_1w[i-1])
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume confirmation: 20-period average
    volume_1d = df_1d['volume'].values
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align weekly and daily indicators to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above daily EMA34, strong volume
            if close[i] > R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below daily EMA34, strong volume
            elif close[i] < S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below daily EMA34
            if close[i] < S1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or above daily EMA34
            if close[i] > R1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals