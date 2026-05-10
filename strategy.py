#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Breakouts from Camarilla R3/S3 levels on 6h with 1w trend filter (EMA34) and volume confirmation.
# Uses weekly EMA34 to filter trend direction (avoiding counter-trend trades) and volume to confirm breakout strength.
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
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Camarilla levels (based on previous day)
    def calculate_camarilla(h, l, c):
        # Typical price for the day
        typical = (h + l + c) / 3.0
        range_ = h - l
        # Camarilla levels
        R3 = c + (range_ * 1.1000 / 4)
        S3 = c - (range_ * 1.1000 / 4)
        return R3, S3
    
    R3_1d = np.full_like(close_1d, np.nan)
    S3_1d = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        R3_1d[i], S3_1d[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Volume confirmation: 20-period average on 1d
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align all indicators to lower timeframe (wait for higher timeframe bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(R3_aligned[i]) or \
           np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above weekly EMA34, strong volume
            if close[i] > R3_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, below weekly EMA34, strong volume
            elif close[i] < S3_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 or below weekly EMA34
            if close[i] < S3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or above weekly EMA34
            if close[i] > R3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals