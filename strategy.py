#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_20_Volume_Trend
# Hypothesis: 4-hour breakouts from 20-period Donchian channels with volume confirmation and trend filter (EMA20). 
# Donchian breakouts capture momentum; volume ensures breakout strength; EMA20 filter avoids counter-trend trades.
# Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Donchian_Breakout_20_20_Volume_Trend"
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
    
    # 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    upper = rolling_max(high, 20)
    lower = rolling_min(low, 20)
    
    # 4h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h volume MA (20-period)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # 1h EMA50 for higher timeframe trend filter (optional but improves robustness)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i]) or \
           np.isnan(ema_50_1h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, above EMA20, strong volume, and above 1h EMA50
            if close[i] > upper[i] and close[i] > ema_20[i] and volume[i] > 1.5 * vol_ma_20[i] and close[i] > ema_50_1h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below EMA20, strong volume, and below 1h EMA50
            elif close[i] < lower[i] and close[i] < ema_20[i] and volume[i] > 1.5 * vol_ma_20[i] and close[i] < ema_50_1h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below lower Donchian or below EMA20
            if close[i] < lower[i] or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian or above EMA20
            if close[i] > upper[i] or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals