#!/usr/bin/env python3
# 12h_KAMA_Trend_1wTrend_Volume
# Hypothesis: 12-hour entries in direction of weekly KAMA trend with volume confirmation.
# Weekly KAMA filters trend to avoid counter-trend trades. Entry when price crosses above/below
# 12-period KAMA on 12h chart with volume > 1.5x 20-period average. Designed for 12h to achieve
# 12-37 trades/year, suitable for both bull and bear markets by following the higher timeframe trend.

name = "12h_KAMA_Trend_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_len=10, fast_len=2, slow_len=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i-1] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly KAMA for trend filter
    kama_1w = kama(close_1w, er_len=10, fast_len=2, slow_len=30)
    
    # 12h KAMA for entry signal
    kama_12h = kama(close, er_len=10, fast_len=2, slow_len=30)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align weekly KAMA to 12h timeframe (wait for 1w bar to close)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(kama_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 12h KAMA, above weekly KAMA, strong volume
            if close[i] > kama_12h[i] and close[i] > kama_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below 12h KAMA, below weekly KAMA, strong volume
            elif close[i] < kama_12h[i] and close[i] < kama_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below 12h KAMA or below weekly KAMA
            if close[i] < kama_12h[i] or close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above 12h KAMA or above weekly KAMA
            if close[i] > kama_12h[i] or close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals