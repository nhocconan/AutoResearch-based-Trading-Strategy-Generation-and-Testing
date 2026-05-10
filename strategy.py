#!/usr/bin/env python3
# 6h_SR_Reversal_Volume_1dTrend
# Hypothesis: 6-hour mean reversion at daily support/resistance levels with daily trend filter and volume confirmation.
# In ranging markets (2025-2026), price tends to revert from daily S1/S2/R1/R2 levels.
# Daily EMA50 filter ensures trades align with higher timeframe trend to avoid chop.
# Volume spike confirms rejection at levels. Designed for 6h to achieve 12-30 trades/year.

name = "6h_SR_Reversal_Volume_1dTrend"
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
    
    # Daily data for support/resistance and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily support/resistance levels (pivot-based)
    def calculate_sr(h, l, c):
        pivot = (h + l + c) / 3.0
        range_ = h - l
        S1 = pivot - range_
        S2 = pivot - 2 * range_
        R1 = pivot + range_
        R2 = pivot + 2 * range_
        return S1, S2, R1, R2
    
    S1 = np.full_like(close_1d, np.nan)
    S2 = np.full_like(close_1d, np.nan)
    R1 = np.full_like(close_1d, np.nan)
    R2 = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        S1[i], S2[i], R1[i], R2[i] = calculate_sr(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 6h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or \
           np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near S1/S2 with rejection, above daily EMA50, strong volume
            near_support = (low[i] <= S1_aligned[i] * 1.002 and low[i] >= S2_aligned[i] * 0.998) or \
                           (low[i] <= S2_aligned[i] * 1.002 and low[i] >= S2_aligned[i] * 0.998)
            if near_support and close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price near R1/R2 with rejection, below daily EMA50, strong volume
            elif near_resistance := ((high[i] >= R1_aligned[i] * 0.998 and high[i] <= R2_aligned[i] * 1.002) or
                                     (high[i] >= R2_aligned[i] * 0.998 and high[i] <= R2_aligned[i] * 1.002)):
                if close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price moves to midpoint or breaks above R1
            midpoint = (S1_aligned[i] + R1_aligned[i]) / 2
            if close[i] >= midpoint or close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves to midpoint or breaks below S1
            midpoint = (S1_aligned[i] + R1_aligned[i]) / 2
            if close[i] <= midpoint or close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals