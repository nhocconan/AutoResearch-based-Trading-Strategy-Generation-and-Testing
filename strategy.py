#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_Volume
# Hypothesis: Use 4h trend (EMA50) as directional filter and 1h Camarilla R3/S3 breakouts with volume confirmation for entry timing.
# This combines higher timeframe trend direction with lower timeframe precision entries to reduce whipsaw.
# Designed for 1h to achieve 15-37 trades/year, suitable for both bull and bear markets by following 4h trend.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from 1d data (based on previous day)
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
    
    # Volume confirmation: 20-period average on 1h data
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(R3_1d_aligned[i]) or \
           np.isnan(S3_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend, price breaks above R3, volume confirmation
            if close[i] > ema_50_4h_aligned[i] and close[i] > R3_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, price breaks below S3, volume confirmation
            elif close[i] < ema_50_4h_aligned[i] and close[i] < S3_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend turns down or price breaks below S3
            if close[i] < ema_50_4h_aligned[i] or close[i] < S3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend turns up or price breaks above R3
            if close[i] > ema_50_4h_aligned[i] or close[i] > R3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals