#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Breakouts from weekly Camarilla R3/S3 levels on 12h with 1d trend filter (EMA34) and volume confirmation.
# Weekly levels provide stronger institutional support/resistance; daily EMA34 filters intermediate trend; volume confirms breakout strength.
# Designed for 12h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Weekly data for stronger Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Camarilla levels (based on previous week)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R3 = c + (range_ * 1.1000 / 4)
        S3 = c - (range_ * 1.1000 / 4)
        return R3, S3
    
    R3_1w = np.full_like(close_1w, np.nan)
    S3_1w = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        R3_1w[i], S3_1w[i] = calculate_camarilla(high_1w[i-1], low_1w[i-1], close_1w[i-1])
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average on weekly volume
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1w = mean_arr(volume_1w, 20)
    
    # Align all indicators to lower timeframe
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, above daily EMA34, strong weekly volume
            if close[i] > R3_1w_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, below daily EMA34, strong weekly volume
            elif close[i] < S3_1w_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below weekly S3 or below daily EMA34
            if close[i] < S3_1w_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above weekly R3 or above daily EMA34
            if close[i] > R3_1w_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals