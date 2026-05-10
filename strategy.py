#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from Camarilla R3/S3 levels on 4h with 1d trend filter (EMA34) and volume confirmation.
# Designed for 4h to achieve 20-50 trades/year, using institutional levels with trend and volume filters to avoid false breakouts.
# Works in bull markets (breakouts with trend) and bear markets (fades from extremes with trend filter).

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # 1d data for Camarilla levels, EMA34 trend, and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (R3, S3) based on previous day
    def calculate_camarilla(h, l, c):
        range_ = h - l
        R3 = c + (range_ * 1.1000 / 4)
        S3 = c - (range_ * 1.1000 / 4)
        return R3, S3
    
    R3 = np.full_like(close_1d, np.nan)
    S3 = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        R3[i], S3[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align all indicators to 4h timeframe (wait for 1d bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above EMA34 (uptrend), strong volume
            if close[i] > R3_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, below EMA34 (downtrend), strong volume
            elif close[i] < S3_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 or below EMA34 (trend change)
            if close[i] < S3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or above EMA34 (trend change)
            if close[i] > R3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals