#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_Volume
# Hypothesis: Breakouts from Camarilla R3/S3 levels on 1h with 4h trend filter (EMA34) and volume confirmation.
# Uses 4h EMA34 for trend direction and 1h volume spike for entry confirmation.
# Designed for 1h to achieve 15-37 trades/year with strict entry criteria.
# Works in both bull and bear markets by following 4h trend direction.

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
    
    # 4h data for Camarilla levels and EMA34 trend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels (based on previous 4h bar)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R3 = c + (range_ * 1.1000 / 4)
        S3 = c - (range_ * 1.1000 / 4)
        return R3, S3
    
    R3_4h = np.full_like(close_4h, np.nan)
    S3_4h = np.full_like(close_4h, np.nan)
    for i in range(1, len(close_4h)):
        R3_4h[i], S3_4h[i] = calculate_camarilla(high_4h[i-1], low_4h[i-1], close_4h[i-1])
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 1h timeframe
    R3_4h_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    S3_4h_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1h volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R3_4h_aligned[i]) or np.isnan(S3_4h_aligned[i]) or \
           np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above 4h EMA34, strong volume
            if close[i] > R3_4h_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, below 4h EMA34, strong volume
            elif close[i] < S3_4h_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 or below 4h EMA34
            if close[i] < S3_4h_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R3 or above 4h EMA34
            if close[i] > R3_4h_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals