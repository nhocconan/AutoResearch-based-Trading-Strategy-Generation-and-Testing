#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_1dTrend_Volume
# Hypothesis: 6-hour breakouts from weekly pivot levels (S3/R3) with daily trend filter (EMA34) and volume confirmation.
# Weekly pivots provide strong support/resistance levels from longer timeframe, daily EMA34 filters trend direction to avoid counter-trend trades,
# Volume confirmation ensures breakout strength. Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets.

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
    
    # Daily data for EMA34 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly pivot points (based on previous week)
    def calculate_pivots(h, l, c):
        pivot = (h + l + c) / 3.0
        range_ = h - l
        S3 = c - (range_ * 1.1000 * 2)  # S3 = C - 2*range*1.1
        R3 = c + (range_ * 1.1000 * 2)  # R3 = C + 2*range*1.1
        return S3, R3
    
    S3_1w = np.full_like(close_1w, np.nan)
    R3_1w = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        S3_1w[i], R3_1w[i] = calculate_pivots(high_1w[i-1], low_1w[i-1], close_1w[i-1])
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align weekly indicators to 6h timeframe (wait for weekly bar to close)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    
    # Align daily indicators to 6h timeframe (wait for daily bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(S3_1w_aligned[i]) or np.isnan(R3_1w_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above daily EMA34, strong volume
            if close[i] > R3_1w_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, below daily EMA34, strong volume
            elif close[i] < S3_1w_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 or below daily EMA34
            if close[i] < S3_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or above daily EMA34
            if close[i] > R3_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals