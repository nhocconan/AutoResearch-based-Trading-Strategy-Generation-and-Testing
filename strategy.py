#!/usr/bin/env python3
# 4h_PivotPoint_Reversal_1dTrend_Volume
# Hypothesis: 4-hour reversals at daily pivot points (S1/S2/R1/R2) with daily trend filter (EMA50) and volume confirmation.
# Daily EMA50 filters trend direction to avoid counter-trend trades; daily pivot levels provide precise reversal zones;
# Volume confirmation ensures rejection strength. Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.
# Uses only price action and volume, no complex oscillators, to minimize overfitting and maximize robustness.

name = "4h_PivotPoint_Reversal_1dTrend_Volume"
timeframe = "4h"
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
    
    # Daily data for EMA50 trend filter and pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pivot points (standard formula) based on previous day
    def calculate_pivot_points(h, l, c):
        pivot = (h + l + c) / 3.0
        range_ = h - l
        R1 = pivot + (range_ * 1.0 / 3.0)
        S1 = pivot - (range_ * 1.0 / 3.0)
        R2 = pivot + (range_ * 2.0 / 3.0)
        S2 = pivot - (range_ * 2.0 / 3.0)
        return pivot, R1, S1, R2, S2
    
    pivot = np.full_like(close_1d, np.nan)
    R1 = np.full_like(close_1d, np.nan)
    S1 = np.full_like(close_1d, np.nan)
    R2 = np.full_like(close_1d, np.nan)
    S2 = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        p, r1, s1, r2, s2 = calculate_pivot_points(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        pivot[i] = p
        R1[i] = r1
        S1[i] = s1
        R2[i] = r2
        S2[i] = s2
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 4h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long reversal: price rejects S1 or S2 with strong volume, above EMA50 (uptrend)
            if ((close[i] > S1_aligned[i] and low[i] <= S1_aligned[i]) or 
                (close[i] > S2_aligned[i] and low[i] <= S2_aligned[i])) and \
               close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R1 or R2 with strong volume, below EMA50 (downtrend)
            elif ((close[i] < R1_aligned[i] and high[i] >= R1_aligned[i]) or 
                  (close[i] < R2_aligned[i] and high[i] >= R2_aligned[i])) and \
                 close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or below EMA50
            if close[i] < S1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or above EMA50
            if close[i] > R1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals