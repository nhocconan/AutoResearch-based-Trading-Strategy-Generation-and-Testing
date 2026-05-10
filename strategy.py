#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_Trend_4hEMA50_VolumeConfirm
# Hypothesis: Breakouts from Donchian(20) channel on 4h with 4h EMA50 trend filter and volume confirmation.
# Donchian channels capture institutional breakout patterns; EMA50 filters trend direction; volume confirms breakout strength.
# Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Donchian_Breakout_20_Trend_4hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian(20), EMA50, and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian(20) channels (based on previous 20 periods)
    def highest(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.max(arr[i - p + 1:i + 1])
        return res
    
    def lowest(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.min(arr[i - p + 1:i + 1])
        return res
    
    upper = highest(high_4h, 20)
    lower = lowest(low_4h, 20)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_4h, 20)
    
    # Align all indicators to lower timeframe (wait for 4h bar to close)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, above EMA50, strong volume
            if close[i] > upper_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below EMA50, strong volume
            elif close[i] < lower_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below lower Donchian or below EMA50
            if close[i] < lower_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian or above EMA50
            if close[i] > upper_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals