#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Breakouts from daily Camarilla R3/S3 levels on 6h with weekly trend filter (EMA50) and volume confirmation.
# Uses weekly EMA50 to filter trend direction (bull/bear), daily Camarilla for intraday S/R, and volume spike for breakout confirmation.
# Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets by aligning with higher timeframe trend.

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Camarilla levels (based on previous day)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R3 = c + (range_ * 1.1000 / 4)
        S3 = c - (range_ * 1.1000 / 4)
        return R3, S3
    
    R3_1d = np.full_like(close_1d, np.nan)
    S3_1d = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        R3_1d[i], S3_1d[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align all indicators to 6h timeframe (wait for weekly/daily bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(R3_1d_aligned[i]) or \
           np.isnan(S3_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above weekly EMA50 (bullish trend), strong volume
            if close[i] > R3_1d_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, below weekly EMA50 (bearish trend), strong volume
            elif close[i] < S3_1d_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 or below weekly EMA50 (trend change)
            if close[i] < S3_1d_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or above weekly EMA50 (trend change)
            if close[i] > R3_1d_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals